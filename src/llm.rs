use crate::tensor::{Tensor, WeightTensor, bf16_u16_to_f32, bytes_to_u16_slice, update_stride};
use std::{time::UNIX_EPOCH, vec};

use memmap::Mmap;
use rayon::{in_place_scope, iter::{IntoParallelIterator, ParallelIterator}};
use serde_json::Value;

use rayon::prelude::*;

const rayon_thresshold:usize = 0;

pub fn get_weight_shape(weight_name: &str, structure_json: &Value) -> Result<Vec<usize>, String> {
    let value = structure_json[weight_name].clone();

    let shape = value["shape"]
        .as_array()
        .expect("cannot extract token shape")
        .iter()
        .map(|x| x.as_u64().expect("cannot convert num to u64.") as usize)
        .collect();

    Ok(shape)
}

pub fn get_weight_matrix<'a>(
    weight_name: &str,
    structure_json: &Value,
    mmap: &'a Mmap,
    header_size: usize,
) -> Result<WeightTensor<'a>, String> {
    let value = &structure_json[weight_name];
    let offset: Vec<usize> = value["data_offsets"]
        .as_array()
        .expect("cannot extract token offset")
        .iter()
        .map(|x| x.as_u64().expect("cannot convert num to u64.") as usize)
        .collect();
    //let dtype = value["dtype"].as_str().expect("cannot extract dtype.");
    let shape = value["shape"]
        .as_array()
        .expect("cannot extract token shape")
        .iter()
        .map(|x| x.as_u64().expect("cannot convert num to u64.") as usize)
        .collect();

    let result_raw = &mmap[8 + header_size as usize + offset[0] as usize
        ..8 + header_size as usize + offset[1] as usize];

    let weight = WeightTensor::new(
        bytes_to_u16_slice(result_raw).expect("cannot convert to &[f32]"),
        shape,
    );

    Ok(weight)
}

pub fn token_embedding(
    token_ids: &Vec<usize>,
    weight_tensor: &WeightTensor,
    x: &mut Tensor,
) -> Result<(), String> {
    let hidden_dim = weight_tensor.shape[1];

    x.data = token_ids
        .iter()
        .flat_map(|&id| {
            weight_tensor.data[(id * hidden_dim)..((id + 1) * hidden_dim)]
                .iter()
                .map(|x| bf16_u16_to_f32(*x))
        })
        .collect();

    x.update_shape(vec![token_ids.len(), hidden_dim]);

    Ok(())
}

pub fn rmsnorm(
    input: &Tensor,
    weight: &WeightTensor,
    hidden_dim: usize,
    epsilon: f32,
) -> Result<Tensor, String> {
    let result: Vec<f32> = input
        .data
        .chunks_exact(hidden_dim)
        .flat_map(|chunk| {
            let accumulator: f32 = chunk.iter().map(|x| x * x).sum();
            let denominator = f32::sqrt((accumulator / hidden_dim as f32) + epsilon);
            chunk
                .iter()
                .zip(weight.data.iter())
                .map(move |(&x, &w)| (x / denominator) * bf16_u16_to_f32(w))
        })
        .collect();

    Ok(Tensor {
        data: result,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    })
}

pub fn linear_proj(
    input: &Tensor,
    weight: &WeightTensor,
    bias: &WeightTensor,
    q: &mut Tensor,
) -> Result<(), String> {
    let input_shape = input.shape.clone();
    let weight_shape = weight.shape.clone();

    //let mut result = vec![0.0; input_shape[0] * weight_shape[0]];

    assert!(input_shape[0] * weight_shape[0] <= q.data.len());
    //q.shape = vec![input_shape[0],weight_shape[0]];
    //q.strides = update_stride(&q.shape).expect("cannot update stride");

    // 3. Perform the triple loop matrix multiplication
    for i in 0..input_shape[0] {
        for j in 0..weight_shape[0] {
            let mut sum = 0.0;

            let x_start = i * input_shape[1];
            let x_end = (i + 1) * input_shape[1];
            let w_start = j * weight_shape[1];
            let w_end = (j + 1) * weight_shape[1];

            let x_slices = &input.data[x_start..x_end];
            let w_slices = &weight.data[w_start..w_end];

            sum = dot_avx2_bf16(x_slices, w_slices);

            // Store the final dot product result
            let out_idx = i * q.strides[0] + j;
            q.data[out_idx] = sum + bf16_u16_to_f32(bias.data[j]);
        }
    }

    Ok(())
}

pub fn apply_rope(tensor: &mut Tensor, rope_theta: f32, current_pos: usize) {
    for pos in 0..tensor.shape[0] {
        for h in 0..tensor.shape[1] {
            for d in 0..tensor.shape[2] / 2 {
                let idx1 =
                    pos * tensor.strides[0] + h * tensor.strides[1] + (d) * tensor.strides[2];
                let idx2 = pos * tensor.strides[0]
                    + h * tensor.strides[1]
                    + (d + (tensor.shape[2] / 2)) * tensor.strides[2];

                let x1 = tensor.data[idx1];
                let x2 = tensor.data[idx2];

                let power: f32 = -2.0 * d as f32 / tensor.shape[2] as f32;
                let wd = rope_theta.powf(power);
                let theta = (pos + current_pos) as f32 * wd;

                let x1n = x1 * f32::cos(theta) - x2 * f32::sin(theta);
                let x2n = x1 * f32::sin(theta) + x2 * f32::cos(theta);

                tensor.data[idx1] = x1n;
                tensor.data[idx2] = x2n;
            }
        }
    }
}

pub fn attention_score(
    q: &Tensor,
    k: &Tensor,
    kv_group: usize,
    valid_kv_len: usize,
    current_pos: usize,
    s: &mut Tensor,
) -> Result<(), String> {
    s.update_shape(vec![q.shape[1], q.shape[0], valid_kv_len]);

    //print!("S shape: {:?} S stride:  {:?} S data size: {:?}",&s.shape, &s.strides, &s.data.len());
    for h in 0..s.shape[0] {
        //iter through head
        for pos_q in 0..s.shape[1] {
            for pos_k in 0..s.shape[2] {
                let idx = h * s.strides[0] + pos_q * s.strides[1] + pos_k * s.strides[2] as usize;
                let abs_pos_q = pos_q + (current_pos - (q.shape[0] - 1));
                s.data[idx] = 0.0;
                if abs_pos_q >= pos_k {
                    //determine use which K head
                    let k_h = h / kv_group as usize;
                    let mut sum = 0.0 as f32;
                    //s.data[idx] = q.data[pos_q][h][d] * k.data[pos_k][h_k][d]
                    //check if both hidden_dim are the same
                    if q.shape[2] != k.shape[2] {
                        return Err("q and k hidden_dim not the same".to_string());
                    }
                    let hidden_dim = q.shape[2];
                    for d in 0..hidden_dim {
                        let idq = pos_q * q.strides[0] + h * q.strides[1] + d * q.strides[2];
                        let idk = pos_k * k.strides[0] + k_h * k.strides[1] + d * k.strides[2];
                        sum += q.data[idq] * k.data[idk];
                    }
                    sum *= 1.0 / f32::sqrt(hidden_dim as f32);
                    s.data[idx] = sum;
                } else {
                    //S.data[pos_q][pos_k]
                    //print!(" {}",idx);
                    s.data[idx] = -f32::INFINITY;
                }
            }
        }
    }

    Ok(())
}

pub fn softmax(t: &mut Tensor) {
    for h in 0..t.shape[0] {
        for pos_q in 0..t.shape[1] {
            let mut max = -f32::INFINITY;
            let start_idx = h * t.strides[0] + pos_q * t.strides[1];
            let mut idx = start_idx;
            for _ in 0..t.shape[2] {
                if t.data[idx] > max && t.data[idx] != -f32::INFINITY {
                    max = t.data[idx];
                }
                idx += t.strides[2];
            }

            let mut sum = 0.0;
            idx = start_idx;
            for _ in 0..t.shape[2] {
                if t.data[idx] == -f32::INFINITY {
                    t.data[idx] = 0.0;
                } else {
                    let exp_value = f32::exp(t.data[idx] - max);
                    t.data[idx] = exp_value;
                    sum += exp_value;
                }
                idx += t.strides[2];
            }

            idx = start_idx;
            for _ in 0..t.shape[2] {
                if t.data[idx] > -f32::INFINITY {
                    t.data[idx] /= sum + 1e-6;
                }
                idx += t.strides[2];
            }
        }
    }
}

pub fn attn_out(s: &Tensor, v: &Tensor, kv_group: usize, attn: &mut Tensor) -> Result<(), String> {
    attn.update_shape(vec![s.shape[1], s.shape[0], v.shape[2]]);

    assert!(
        attn.data.len() >= attn.shape.iter().product(),
        "attn shape wrong"
    );

    for pos_q in 0..attn.shape[0] {
        for h in 0..attn.shape[1] {
            for d in 0..attn.shape[2] {
                let mut sum = 0.0 as f32;
                for pos_k in 0..s.shape[2] {
                    let idx_s = h * s.strides[0] + pos_q * s.strides[1] + pos_k * s.strides[2];
                    let idx_v =
                        pos_k * v.strides[0] + (h / kv_group) * v.strides[1] + d * v.strides[2];
                    sum += s.data[idx_s] * v.data[idx_v];
                }
                let idx = pos_q * attn.strides[0] + h * attn.strides[1] + d * attn.strides[2];
                attn.data[idx] = sum;
            }
        }
    }

    Ok(())
}

pub fn out_proj(attn: &Tensor, o: &WeightTensor, atten_fn: &mut Tensor) -> Result<(), String> {


    if attn.shape[0] > rayon_thresshold {
        out_proj_rayon(attn, o, atten_fn);
    } else {
        out_proj_single(attn, o, atten_fn);
    }

    Ok(())
}

pub fn out_proj_single(attn: &Tensor, o: &WeightTensor, atten_fn: &mut Tensor) -> Result<(), String> {
    atten_fn.update_shape(vec![attn.shape[0], attn.shape[1]]);

    assert!(
        atten_fn.data.len() >= atten_fn.shape.iter().product(),
        "attn shape wrong"
    );
    atten_fn.data.fill(0.0);

    let o0 = o.strides[0];
    let a0 = attn.strides[0];
    let af0 = atten_fn.strides[0];

    for i in 0..attn.shape[0] {
        let i_offset = i * a0;
        let af_offset = i * af0;
        for k in 0..attn.shape[1] {
            let attn_val =attn.data[i_offset + k];
            for j in 0..o.shape[0] {
                atten_fn.data[af_offset + j] += attn_val  * bf16_u16_to_f32(o.data[(j * o0) + k]);
            }
        }
    }

    Ok(())
}

pub fn out_proj_rayon(attn: &Tensor, o: &WeightTensor, atten_fn: &mut Tensor) -> Result<(), String> {
    atten_fn.update_shape(vec![attn.shape[0], attn.shape[1]]);

    assert!(
        atten_fn.data.len() >= atten_fn.shape.iter().product(),
        "attn shape wrong"
    );
    atten_fn.data.fill(0.0);

    let o0 = o.strides[0];
    let a0 = attn.strides[0];

    let o_rows = o.shape[0];
    let k_len = attn.shape[1];

    let total = atten_fn.shape.iter().product();

    atten_fn.data[..total]
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx,out_val)|{
            let i = idx / o_rows;
            let k = idx % o_rows;

            
            let x_row_offset = i * a0;
            let w_row_offset = k * o0;
            
            let x_slices = &attn.data[x_row_offset .. x_row_offset + o.shape[1]];
            let w_slices = &o.data[w_row_offset .. w_row_offset + o.shape[1]];

            *out_val = dot_avx2_bf16(x_slices, w_slices);
        });

    Ok(())
}


pub fn res_conn(x: &mut Tensor, attn_final: &Tensor) {
    //print!("{:?}",x.shape);
    let valid_data: usize = x.shape.iter().product();
    x.data[..valid_data]
        .iter_mut()
        .zip(attn_final.data.iter())
        .for_each(|(a, b)| *a += *b);
}

pub fn mlp_mul(x: &Tensor, weight: &WeightTensor, output: &mut Tensor) -> Result<(), String> {

    if x.shape[0] >= rayon_thresshold {
        mlp_mul_rayon(x, weight, output);
    } else {
        mlp_mul_single(x, weight, output);
    }

Ok(())
}

pub fn mlp_mul_single(x: &Tensor, weight: &WeightTensor, output: &mut Tensor) -> Result<(), String> {
    //print!("{:?} : {:?}", x.shape, weight.shape);
    output.update_shape(vec![x.shape[0], weight.shape[0]]);
    let s1 = x.strides[0];
    let s2 = x.strides[1];
    let w1 = weight.strides[0];
    let w2 = weight.strides[1];
    let o1 = output.strides[0];
    let o2 = output.strides[1];
    let weight_rows = weight.shape[0];
    let weight_col = weight.shape[1];

    
    for i in 0..x.shape[0] {
        for k in 0..weight.shape[0] {
            
            let x_row_offset = i * s1;
            let w_row_offset = k*w1;

            let x_slice = &x.data[x_row_offset .. x_row_offset + weight_col * s2];
            let w_slice = &weight.data[w_row_offset .. w_row_offset + weight_col * w2];
            let sum = dot_avx2_bf16(x_slice, w_slice);
            output.data[i * o1 + k * o2] = sum;
        }
    }

    Ok(())
}

use core::arch::x86_64::*;
pub fn mlp_mul_rayon(x: &Tensor, weight: &WeightTensor, output: &mut Tensor) -> Result<(), String> {
    //print!("{:?} : {:?}", x.shape, weight.shape);
    output.update_shape(vec![x.shape[0], weight.shape[0]]);
    let s1 = x.strides[0];
    let s2 = x.strides[1];
    let w1 = weight.strides[0];
    let w2 = weight.strides[1];
    let weight_rows = weight.shape[0];
    let weight_col = weight.shape[1];

    let total:usize = output.shape.iter().product();

    output.data[..total]
        .par_iter_mut()
        .with_min_len(5)
        .enumerate()
        .for_each(|(idx,out_val)|{
            let i = idx / weight_rows;
            let k = idx % weight_rows;
            
            let x_row_offset = i * s1;
            let w_row_offset = k*w1;

            let mut sum: f32 = 0.0;

            let x_slice = &x.data[x_row_offset .. x_row_offset + weight_col * s2];
            let w_slice = &weight.data[w_row_offset .. w_row_offset + weight_col * w2];



            sum = dot_avx2_bf16(x_slice, w_slice);

            
            *out_val = sum;
        });

    Ok(())
}

fn dot_avx2_bf16(x:&[f32], w:&[u16]) -> f32 {
    let mut sum = 0.0;
                //let prefetch_dis = 32;
    unsafe{
                let mut sum_vec = _mm256_setzero_ps();

                let mut x_chunks = x.chunks_exact(8);
                let mut w_chunks = w.chunks_exact(8);

                for (x_chunk,w_chunk) in x_chunks.by_ref().zip(w_chunks.by_ref()) {
                    let x_ptr = x_chunk.as_ptr();
                    let x_vec = _mm256_loadu_ps(x_ptr);

                    let w_ptr = w_chunk.as_ptr();
                    let w_128 = _mm_loadu_si128(w_ptr as *const __m128i);

                    //here do prefetch
                    //_mm_prefetch::<_MM_HINT_T0>(x_ptr.add(prefetch_dis) as *const i8);

                    //_mm_prefetch::<_MM_HINT_NTA>(w_ptr.add(prefetch_dis) as *const i8);

                    let w_256_int = _mm256_cvtepu16_epi32(w_128);

                    let w_256_shifted = _mm256_slli_epi32(w_256_int, 16);

                    let w_vec = _mm256_castsi256_ps(w_256_shifted);

                    sum_vec = _mm256_fmadd_ps(x_vec, w_vec, sum_vec);

                }
                let mut arr = [0.0f32;8];
                _mm256_storeu_ps(arr.as_mut_ptr(),sum_vec);

                sum = arr.iter().sum();

                let x_rem = x_chunks.remainder();
                let w_rem = w_chunks.remainder();
                for i in 0..x_rem.len() {
                    let x_val = x_rem[i];
                    let w_val = bf16_u16_to_f32(w_rem[i]);
                    sum += x_val * w_val;
                }

            }

            sum
}

pub fn silu(weight: &mut Tensor, up: &Tensor) {
    weight
        .data
        .iter_mut()
        .zip(up.data.iter())
        .for_each(|(x, u)| *x = (*x / (1.0 + f32::exp(-*x))) * *u);
}

pub fn random() -> Result<f32, String> {
    let time = std::time::SystemTime::now();
    let mut x: u32 = time
        .duration_since(UNIX_EPOCH)
        .expect("cannot get current time")
        .as_nanos() as u32;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    Ok((x as f32) / (u32::MAX as f32))
}

//test [Generate by Gemini :D]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use std::vec;

    fn mock_f32_to_bf16(val: f32) -> u16 {
        (val.to_bits() >> 16) as u16
    }

    // 物理级探针：处理 f32 浮点数计算时的微小精度误差
    macro_rules! assert_f32_eq {
        ($a:expr, $b:expr) => {
            assert!(
                ($a - $b).abs() < 1e-4,
                "精度校验失败! 物理内存值左侧: {}, 右侧: {}",
                $a,
                $b
            );
        };
    }

    #[test]
    fn test_token_embedding() {
        let token_ids = vec![1]; // 选取词表第 2 个 token
        let raw_f32 = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let w_data = raw_f32
            .iter()
            .map(|&x| mock_f32_to_bf16(x))
            .collect::<Vec<u16>>();
        let weight_tensor = WeightTensor {
            data: &w_data,
            shape: vec![2, 3], // 2个token，hidden_dim=3
            strides: vec![3, 1],
        };

        let mut out = Tensor::new(vec![0.0; 2 * 3], vec![0]);
        token_embedding(&token_ids, &weight_tensor, &mut out).expect("embedding 提取崩溃");
        assert_eq!(out.shape, vec![1, 3]);
        assert_f32_eq!(out.data[0], 0.4);
        assert_f32_eq!(out.data[1], 0.5);
        assert_f32_eq!(out.data[2], 0.6);
    }

    #[test]
    fn test_rmsnorm_math() {
        let x = Tensor {
            data: vec![3.0, 4.0],
            shape: vec![2],
            strides: vec![1],
        };
        let raw_f32 = vec![1.0, 2.0];
        let w_data = raw_f32
            .iter()
            .map(|&x| mock_f32_to_bf16(x))
            .collect::<Vec<u16>>();
        let weight = WeightTensor {
            data: &w_data,
            shape: vec![2],
            strides: vec![1],
        };
        let eps = 1e-5;
        // mean_sq = 12.5, rrms ≈ 0.2828426
        // out = [3*0.2828*1, 4*0.2828*2] = [0.8485, 2.2627]
        let out = rmsnorm(&x, &weight, 2, eps).expect("rmsnorm 计算图崩溃");
        assert_f32_eq!(out.data[0], 0.848528);
        assert_f32_eq!(out.data[1], 2.262741);
    }

    #[test]
    fn test_linear_proj_math() {
        let x = Tensor {
            data: vec![1.0, 2.0],
            shape: vec![1, 2],
            strides: vec![2, 1],
        };
        let raw_f32 = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let w_data = raw_f32
            .iter()
            .map(|&x| mock_f32_to_bf16(x))
            .collect::<Vec<u16>>();
        let w = WeightTensor {
            data: &w_data,
            shape: vec![3, 2],
            strides: vec![2, 1],
        };
        let raw_f32 = vec![0.1, 0.1, 0.1];
        let b_data = raw_f32
            .iter()
            .map(|&x| mock_f32_to_bf16(x))
            .collect::<Vec<u16>>();
        let b = WeightTensor {
            data: &b_data, // bias
            shape: vec![3],
            strides: vec![1],
        };

        let mut out = Tensor::new(vec![0.0; 3], vec![0]);

        linear_proj(&x, &w, &b, &mut out).expect("linear_proj 计算图崩溃");
        //assert_eq!(out.shape, vec![1, 3]);
        // out[0] = 1*0.1 + 2*0.2 + 0.1 = 0.6
        // out[1] = 1*0.3 + 2*0.4 + 0.1 = 1.2
        assert_f32_eq!(out.data[0], 0.6);
        assert_f32_eq!(out.data[1], 1.2);
    }

    #[test]
    fn test_apply_rope() {
        let mut t = Tensor {
            data: vec![1.0, 1.0], // shape [1, 1, 2]
            shape: vec![1, 1, 2],
            strides: vec![2, 2, 1],
        };
        // 若 pos=0，theta=0，cos=1, sin=0 -> 数据不变
        apply_rope(&mut t, 10000.0, 0);
        assert_f32_eq!(t.data[0], 1.0);
        assert_f32_eq!(t.data[1], 1.0);
    }

    #[test]
    fn test_attention_score() {
        let q = Tensor {
            data: vec![1.0, 2.0],
            shape: vec![1, 1, 2], // pos_q=1, h=1, d=2
            strides: vec![2, 2, 1],
        };
        let k = Tensor {
            data: vec![2.0, 3.0],
            shape: vec![1, 1, 2], // pos_k=1, h=1, d=2
            strides: vec![2, 2, 1],
        };

        let mut out = Tensor::new(vec![0.0; 100], vec![0]);

        // 注意：内部会调用 update_stride(&s.shape)，假设其工作正常
        attention_score(&q, &k, 1, 1, 0, &mut out).unwrap();
        // dot product = 2.0 + 6.0 = 8.0
        // sum *= 1.0 / sqrt(2) ≈ 5.65685
        assert_f32_eq!(out.data[0], 5.65685);
    }

    #[test]
    fn test_softmax() {
        let mut t = Tensor {
            data: vec![0.0, 1.0], // shape [1, 1, 2]
            shape: vec![1, 1, 2],
            strides: vec![2, 2, 1],
        };
        softmax(&mut t);
        // exp(0)/(exp(0)+exp(1)) = 1 / (1 + 2.718) ≈ 0.2689
        // exp(1)/(exp(0)+exp(1)) ≈ 0.7310
        assert_f32_eq!(t.data[0], 0.26894);
        assert_f32_eq!(t.data[1], 0.73105);
    }

    #[test]
    fn test_attn_out() {
        let s = Tensor {
            data: vec![0.5, 0.5], // shape [1, 1, 2] - pos_q, h, pos_k
            shape: vec![1, 1, 2],
            strides: vec![2, 2, 1],
        };
        let v = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0], // shape [2, 1, 2] - pos_k, h, d
            shape: vec![2, 1, 2],
            strides: vec![2, 2, 1],
        };
        // out[d=0] = 0.5*1.0 + 0.5*3.0 = 2.0
        // out[d=1] = 0.5*2.0 + 0.5*4.0 = 3.0
        let mut out = Tensor::new(vec![0.0; 100], vec![0]);
        attn_out(&s, &v, 1, &mut out).unwrap();
        assert_f32_eq!(out.data[0], 2.0);
        assert_f32_eq!(out.data[1], 3.0);
    }

    #[test]
    fn test_out_proj() {
        // 1. 物理重现：构建 attn_out 刚吐出来的 3D 张量 [seq_len, num_heads, head_dim]
        // 假设 seq_len=1, num_heads=2, head_dim=2 (所以总的 hidden_dim = 4)
        let mut attn = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![1, 2, 2],
            strides: vec![4, 2, 1], // 原汁原味的 3D 步长
        };

        // 2. 模拟架构师的硬核暴改 (Zero-Cost In-place Reshape)
        // 直接在堆栈上修改元数据，此时 shape 变为 [1, 4]
        attn.shape = vec![attn.shape[0], attn.shape[1] * attn.shape[2]];

        // 3. 构建 o_proj 权重张量 [out_dim, hidden_dim]
        // 假设我们要把它投射回一个 dim=2 的空间，所以 shape 是 [2, 4]
        let raw_f32 = vec![
            0.1, 0.2, 0.3, 0.4, // 对应第 1 个输出特征
            0.5, 0.6, 0.7, 0.8, // 对应第 2 个输出特征
            0.1, 0.2, 0.3, 0.4, // 对应第 3 个输出特征
            0.5, 0.6, 0.7, 0.8, // 对应第 4 个输出特征
        ];
        let o_data = raw_f32
            .iter()
            .map(|&x| mock_f32_to_bf16(x))
            .collect::<Vec<u16>>();
        let o_proj = WeightTensor {
            data: &o_data,
            shape: vec![4, 4],
            strides: vec![4, 1], // 2D 步长
        };

        // 4. 送入你的原始 out_proj
        let mut out = Tensor::new(vec![0.0; 100], vec![0]);
        out_proj(&attn, &o_proj, &mut out).expect("out_proj 计算图崩溃");

        // 5. 物理期望校验
        // out[0] = 1*0.1 + 2*0.2 + 3*0.3 + 4*0.4 = 0.1 + 0.4 + 0.9 + 1.6 = 3.0
        // out[1] = 1*0.5 + 2*0.6 + 3*0.7 + 4*0.8 = 0.5 + 1.2 + 2.1 + 3.2 = 7.0
        assert_eq!(out.shape, vec![1, 4], "Shape 投射后维度不匹配");
        assert_f32_eq!(out.data[0], 3.0);
        assert_f32_eq!(out.data[1], 7.0);
    }

    #[test]
    fn test_res_conn() {
        let mut x = Tensor {
            data: vec![1.0, 2.0],
            shape: vec![2],
            strides: vec![1],
        };
        let attn = Tensor {
            data: vec![0.5, 0.5],
            shape: vec![2],
            strides: vec![1],
        };
        res_conn(&mut x, &attn);
        assert_f32_eq!(x.data[0], 1.5);
        assert_f32_eq!(x.data[1], 2.5);
    }

    #[test]
    fn test_mlp_mul() {
        let x = Tensor {
            data: vec![1.0, 2.0],
            shape: vec![1, 2],
            strides: vec![2, 1],
        };
        let raw_f32 = vec![2.0, 3.0, 4.0, 5.0];
        let w_data = raw_f32
            .iter()
            .map(|&x| mock_f32_to_bf16(x))
            .collect::<Vec<u16>>();
        let w = WeightTensor {
            data: &w_data, // shape [2, 2]
            shape: vec![2, 2],
            strides: vec![2, 1],
        };
        let mut out = Tensor::new(vec![0.0; 10000], vec![0]);
        // out[0,0] = x[0,0]*w[0,0] + x[0,1]*w[0,1] = 1*2 + 2*3 = 8.0
        // out[0,1] = x[0,0]*w[1,0] + x[0,1]*w[1,1] = 1*4 + 2*5 = 14.0
        mlp_mul(&x, &w, &mut out).unwrap();
        assert_f32_eq!(out.data[0], 8.0);
        assert_f32_eq!(out.data[1], 14.0);
    }

    #[test]
    fn test_silu() {
        let mut w = Tensor {
            data: vec![1.0],
            shape: vec![1],
            strides: vec![1],
        };
        let u = Tensor {
            data: vec![2.0],
            shape: vec![1],
            strides: vec![1],
        };
        // silu(1.0) * 2.0 = (1 / (1+e^-1)) * 2 = 0.73105 * 2 = 1.4621
        silu(&mut w, &u);
        assert_f32_eq!(w.data[0], 1.4621);
    }

    #[test]
    fn test_random_generation() {
        // 仅仅测试物理随机数生成器是否能安全吐出有效区间值，不越界
        let r = random().expect("PRNG 崩溃");
        assert!(r >= 0.0 && r < 1.0, "随机数跳出概率空间");
    }
}
