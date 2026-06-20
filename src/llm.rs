use crate::tensor::{BlockQ8_0, Tensor, WeightData, WeightTensor, bytes_to_f32_slice, bf16_u16_to_f32, bytes_to_q8_slice, bytes_to_u16_slice};
use std::{time::UNIX_EPOCH, vec};

use memmap::Mmap;
use rayon::iter::ParallelIterator;
use serde_json::Value;

use rayon::prelude::*;

const RAYON_THRESHOLD: usize = 0;

/*
pub fn get_weight_shape(weight_name: &str, structure_json: &Value) -> Result<Vec<usize>, String> {
    let value = structure_json[weight_name].clone();

    let shape = value["shape"]
        .as_array()
        .expect("cannot extract token shape")
        .iter()
        .map(|x| x.as_u64().expect("cannot convert num to u64.") as usize)
        .collect();

    Ok(shape)
}*/

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
    let dtype = value["dtype"].as_str().expect("cannot extract dtype.");
    let shape = value["shape"]
        .as_array()
        .expect("cannot extract token shape")
        .iter()
        .map(|x| x.as_u64().expect("cannot convert num to u64.") as usize)
        .collect();

    let result_raw = &mmap[8 + header_size as usize + offset[0] as usize
        ..8 + header_size as usize + offset[1] as usize];

    //println!("{:?}",dtype);

    let weight = if dtype == "BF16" {WeightTensor::new(
            //bytes_to_u16_slice(result_raw).expect("cannot convert to &[f32]"),
            WeightData::BF16(bytes_to_u16_slice(result_raw).expect("cannot convert to &[f32]")),
            shape,
        )
    } else {

        if shape.len() == 2{
            WeightTensor::new(
                        //bytes_to_u16_slice(result_raw).expect("cannot convert to &[f32]"),
                        WeightData::Q8(bytes_to_q8_slice(result_raw).expect("cannot convert to &[q8]")),
                        shape,
                    )
        } else {
            WeightTensor::new(
                //bytes_to_u16_slice(result_raw).expect("cannot convert to &[f32]"),
                WeightData::F32(bytes_to_f32_slice(result_raw).expect("cannot convert to &[q8]")),
                shape,
            )
        }

    };

    Ok(weight)
}

pub fn token_embedding(
    token_ids: &Vec<usize>,
    weight_tensor: &WeightTensor,
    x: &mut Tensor,
) -> Result<(), String> {
    let hidden_dim = weight_tensor.shape[1];
    x.data.clear();

    match &weight_tensor.data {
        WeightData::BF16(bf16_w) => {
            for &id in token_ids{
                let start = id * hidden_dim;
                let end = (id + 1 ) * hidden_dim;
                for &val in &bf16_w[start .. end] {
                    x.data.push(bf16_u16_to_f32(val));
                }
            }
        }
        WeightData::Q8(q8_w) => {
            //This mean to one hidden dim consist of how many q8 block
            let blocks_per_row = hidden_dim / 32;

            for &id in token_ids{
                //find the start idx by mul id (which means the num ber of hidden dim) and the (block consist of a hidden dim)
                let start = id * blocks_per_row;
                let end = start + blocks_per_row;

                for block in &q8_w[start..end] {
                    let scale = block.d;
                    for &q in &block.qs {
                        x.data.push(q as f32 * scale);
                    }
                }

            }
        }
        WeightData::F32(_) => return Err("cannot be f32 weight".to_string())
    }
    x.update_shape(vec![token_ids.len(), hidden_dim]);

    Ok(())
}


pub fn rmsnorm(
    input: &Tensor,
    weight: &WeightTensor,
    hidden_dim: usize,
    epsilon: f32,
) -> Result<Tensor, String> {
 match &weight.data {
        WeightData::BF16(_) => rmsnorm_bf16(input, weight, hidden_dim, epsilon),
        WeightData::Q8(_) => rmsnorm_q8(input, weight, hidden_dim, epsilon),
        WeightData::F32(_) => rmsnorm_f32(input, weight, hidden_dim, epsilon),
    }
}

pub fn rmsnorm_f32(
    input: &Tensor,
    weight: &WeightTensor,
    hidden_dim: usize,
    epsilon: f32,
) -> Result<Tensor, String> {

    let mut result = vec![0.0;input.data.len()];
    let input_row = input.data.chunks_exact(hidden_dim);
    let res_row = result.chunks_exact_mut(hidden_dim);

    for (input, output) in input_row.zip(res_row) {
        let mut accumulator :f32;
        unsafe{
            let mut acc_vec = _mm256_setzero_ps();
            let mut data_chunks = input.chunks_exact(8);
            for c_chunk in data_chunks.by_ref(){
                let data_ptr = c_chunk.as_ptr();
                let data_vec = _mm256_loadu_ps(data_ptr);
                let data_vec_2 = _mm256_loadu_ps(data_ptr);
                acc_vec = _mm256_fmadd_ps(data_vec, data_vec_2, acc_vec);
            }
            let low_128 = _mm256_castps256_ps128(acc_vec);
            let high_128 = _mm256_extractf128_ps(acc_vec,1);

            let mut sum_128 = _mm_add_ps(low_128, high_128);
            let shuf_128 = _mm_shuffle_ps(sum_128, sum_128, 0b01_00_11_10);
            sum_128 = _mm_add_ps(sum_128,shuf_128);

            let shuf_128 = _mm_shuffle_ps(sum_128, sum_128, 0b00_01_00_01);
            sum_128 = _mm_add_ps(sum_128,shuf_128);

            accumulator = _mm_cvtss_f32(sum_128);

            let data_rem  = data_chunks.remainder();
            for data in 0..data_rem.len(){
                let data = data_rem[data];
                accumulator += data * data;
            }

            //denomitor
            let inv_denominator = 1.0 / f32::sqrt((accumulator / hidden_dim as f32) + epsilon);
            let invdenominator_vec =  _mm256_set1_ps(inv_denominator);

            let mut x_chunks = input.chunks_exact(8);
            let weight_data = match &weight.data {
                WeightData::BF16(_) => panic!("cannot use bf16 in bf16 simd"),
                WeightData::Q8(_) => panic!("cannot use q8 in bf16 simd"),
                WeightData::F32(f32) => f32,
            };
            let mut weight_chunks = weight_data.chunks_exact(8);
            let mut out_chunks = output.chunks_exact_mut(8);

            for ((x_chunk,w_chunk),out_chunk) in (x_chunks.by_ref().zip(weight_chunks.by_ref())).zip(out_chunks.by_ref()) {
                let x_ptr = x_chunk.as_ptr();
                let x_vec = _mm256_loadu_ps(x_ptr);

                let w_ptr = w_chunk.as_ptr();
                let w_vec = _mm256_loadu_ps(w_ptr);

                let mut res_vec =  _mm256_mul_ps(x_vec,invdenominator_vec);
                res_vec = _mm256_mul_ps(w_vec, res_vec);

                let out_ptr = out_chunk.as_mut_ptr();
                _mm256_storeu_ps(out_ptr,res_vec);
            }   
            let x_rem = x_chunks.remainder();
            let w_rem = weight_chunks.remainder();
            let out_rem = out_chunks.into_remainder();
            for i in 0..x_rem.len() 
            {
                out_rem[i] = (x_rem[i] * inv_denominator) * w_rem[i];
            }
        }
    }

    Ok(Tensor {
        data: result,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    })
}

pub fn rmsnorm_bf16(
    input: &Tensor,
    weight: &WeightTensor,
    hidden_dim: usize,
    epsilon: f32,
) -> Result<Tensor, String> {

    let mut result = vec![0.0;input.data.len()];
    let input_row = input.data.chunks_exact(hidden_dim);
    let res_row = result.chunks_exact_mut(hidden_dim);

    for (input, output) in input_row.zip(res_row) {
        let mut accumulator :f32;
        unsafe{
            let mut acc_vec = _mm256_setzero_ps();
            let mut data_chunks = input.chunks_exact(8);
            for c_chunk in data_chunks.by_ref(){
                let data_ptr = c_chunk.as_ptr();
                let data_vec = _mm256_loadu_ps(data_ptr);
                let data_vec_2 = _mm256_loadu_ps(data_ptr);
                acc_vec = _mm256_fmadd_ps(data_vec, data_vec_2, acc_vec);
            }
            let low_128 = _mm256_castps256_ps128(acc_vec);
            let high_128 = _mm256_extractf128_ps(acc_vec,1);

            let mut sum_128 = _mm_add_ps(low_128, high_128);
            let shuf_128 = _mm_shuffle_ps(sum_128, sum_128, 0b01_00_11_10);
            sum_128 = _mm_add_ps(sum_128,shuf_128);

            let shuf_128 = _mm_shuffle_ps(sum_128, sum_128, 0b00_01_00_01);
            sum_128 = _mm_add_ps(sum_128,shuf_128);

            accumulator = _mm_cvtss_f32(sum_128);

            let data_rem  = data_chunks.remainder();
            for data in 0..data_rem.len(){
                let data = data_rem[data];
                accumulator += data * data;
            }

            //denomitor
            let inv_denominator = 1.0 / f32::sqrt((accumulator / hidden_dim as f32) + epsilon);
            let invdenominator_vec =  _mm256_set1_ps(inv_denominator);

            let mut x_chunks = input.chunks_exact(8);
            let weight_data = match &weight.data {
                WeightData::BF16(bf16) => bf16,
                WeightData::Q8(_) => panic!("cannot use q8 in bf16 simd"),
                WeightData::F32(_) => panic!("cannot use f32 in bf16 simd")
            };
            let mut weight_chunks = weight_data.chunks_exact(8);
            let mut out_chunks = output.chunks_exact_mut(8);

            for ((x_chunk,w_chunk),out_chunk) in (x_chunks.by_ref().zip(weight_chunks.by_ref())).zip(out_chunks.by_ref()) {
                let x_ptr = x_chunk.as_ptr();
                let x_vec = _mm256_loadu_ps(x_ptr);

                let w_ptr = w_chunk.as_ptr();
                let w_128 = _mm_loadu_si128(w_ptr as *const __m128i);
                let w_256_int = _mm256_cvtepu16_epi32(w_128);
                let w_256_shifted = _mm256_slli_epi32(w_256_int, 16);
                let w_vec = _mm256_castsi256_ps(w_256_shifted);

                let mut res_vec =  _mm256_mul_ps(x_vec,invdenominator_vec);
                res_vec = _mm256_mul_ps(w_vec, res_vec);

                let out_ptr = out_chunk.as_mut_ptr();
                _mm256_storeu_ps(out_ptr,res_vec);
            }   
            let x_rem = x_chunks.remainder();
            let w_rem = weight_chunks.remainder();
            let out_rem = out_chunks.into_remainder();
            for i in 0..x_rem.len() 
            {
                out_rem[i] = (x_rem[i] * inv_denominator) * bf16_u16_to_f32(w_rem[i]);
            }
        }
    }

    Ok(Tensor {
        data: result,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    })
}


pub fn rmsnorm_q8(
    input: &Tensor,
    weight: &WeightTensor,
    hidden_dim: usize,
    epsilon: f32,
) -> Result<Tensor, String> {

    let mut result = vec![0.0;input.data.len()];
    let input_row = input.data.chunks_exact(hidden_dim);
    let res_row = result.chunks_exact_mut(hidden_dim);

    let q8_blocks = match &weight.data{
        WeightData::Q8(blocks) => blocks,
        WeightData::BF16(_) => panic!("cannot use bf16 in q8 simd"),
        WeightData::F32(_) => panic!("cannot use f32 in bf16 simd")
    };

    for (input, output) in input_row.zip(res_row) {
        let mut accumulator :f32;
        unsafe{
            let mut acc_vec = _mm256_setzero_ps();
            let mut data_chunks = input.chunks_exact(8);
            for c_chunk in data_chunks.by_ref(){
                let data_ptr = c_chunk.as_ptr();
                let data_vec = _mm256_loadu_ps(data_ptr);
                let data_vec_2 = _mm256_loadu_ps(data_ptr);
                acc_vec = _mm256_fmadd_ps(data_vec, data_vec_2, acc_vec);
            }
            let low_128 = _mm256_castps256_ps128(acc_vec);
            let high_128 = _mm256_extractf128_ps(acc_vec,1);

            let mut sum_128 = _mm_add_ps(low_128, high_128);
            let shuf_128 = _mm_shuffle_ps(sum_128, sum_128, 0b01_00_11_10);
            sum_128 = _mm_add_ps(sum_128,shuf_128);

            let shuf_128 = _mm_shuffle_ps(sum_128, sum_128, 0b00_01_00_01);
            sum_128 = _mm_add_ps(sum_128,shuf_128);

            accumulator = _mm_cvtss_f32(sum_128);

            let data_rem  = data_chunks.remainder();
            for data in 0..data_rem.len(){
                let data = data_rem[data];
                accumulator += data * data;
            }

            //denomitor
            let inv_denominator = 1.0 / f32::sqrt((accumulator / hidden_dim as f32) + epsilon);
            let invdenominator_vec =  _mm256_set1_ps(inv_denominator);

            let mut x_chunks = input.chunks_exact(32);
            let mut out_chunks = output.chunks_exact_mut(32);
            let w_blocks = q8_blocks.iter();

            for ((x_chunk,out_chunk),block) in (x_chunks.by_ref().zip(out_chunks.by_ref())).zip(w_blocks) {

                let v_scale = _mm256_set1_ps(block.d);
                let q_ptr = block.qs.as_ptr();
                let x_ptr = x_chunk.as_ptr();
                let out_ptr = out_chunk.as_mut_ptr();

                for j in 0..4 {
                    let offset = j * 8;

                    let v_x = _mm256_loadu_ps(x_ptr.add(offset));
                    let v_x_norm = _mm256_mul_ps(v_x,invdenominator_vec);

                    let q_chunk = _mm_loadl_epi64(q_ptr.add(offset) as *const __m128i);
                    let v_q_i32 = _mm256_cvtepi8_epi32(q_chunk);
                    let v_q_f32 = _mm256_cvtepi32_ps(v_q_i32);

                    let v_w = _mm256_mul_ps(v_q_f32 , v_scale);
                    let v_res = _mm256_mul_ps(v_x_norm, v_w);
                    _mm256_storeu_ps(out_ptr.add(offset), v_res);


                }
            }   
        }
    }

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
    let row = input.shape[0];
    let in_f = input.shape[1];
    let out_f = weight.shape[0];
    let weight_shape = weight.shape.clone();

    match &weight.data {
        WeightData::BF16(bf16_w) => {
            let WeightData::BF16(bias_slice) = &bias.data else{ panic!("cannot use bf16");};

            for i in 0..row {
                let start = i * in_f;
                let slices = &input.data[start .. start + in_f];

                for j in 0.. out_f{
                    let w_start = j * in_f;
                    let w_slices = &bf16_w[w_start .. w_start + in_f];

                    let sum = dot_avx2_bf16(slices, w_slices);

                    let out_idx = i * q.strides[0] + j;
                    q.data[out_idx] = sum + bf16_u16_to_f32(bias_slice[j]);
                }

            }   
        }
        WeightData::Q8(q8_weight) => {
            let WeightData::F32(bias_slice) = &bias.data else {
                panic!("bias must be f32 slice");
            };

            let blocks_per_row = in_f / 32;

            for i in 0..row {
                let start = i * in_f;
                let slices = &input.data[start .. start + in_f];

                for j in 0.. out_f{
                    let block_start = j * blocks_per_row;
                    let  block_end = block_start + blocks_per_row;
                    let w_blocks = &q8_weight[block_start .. block_end];

                    let sum = dot_avx2_q8(slices, w_blocks);

                    let out_idx = i * q.strides[0] + j;
                    q.data[out_idx] = sum + bias_slice[j];
                }
            }
        }
        WeightData::F32(_) => {
            panic!("cannot process f32 type of weight");
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
                //let abs_pos_q = pos_q + current_pos;
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
    if attn.shape[0] >= RAYON_THRESHOLD {
        let _ = out_proj_rayon(attn, o, atten_fn);
    } else {
        let _ = out_proj_single(attn, o, atten_fn);
    }

    Ok(())
}

pub fn out_proj_single(
    attn: &Tensor,
    o: &WeightTensor,
    atten_fn: &mut Tensor,
) -> Result<(), String> {
    atten_fn.update_shape(vec![attn.shape[0], attn.shape[1]]);

    assert!(
        atten_fn.data.len() >= atten_fn.shape.iter().product(),
        "attn shape wrong"
    );
    atten_fn.data.fill(0.0);

    let o0 = o.strides[0];
    let a0 = attn.strides[0];
    let af0 = atten_fn.strides[0];

    let in_feature = attn.shape[1];
    let out_feature = o.shape[0];

    match &o.data {
        WeightData::BF16(bf16_slice) => {
            for i in 0..attn.shape[0] {
                let x_start = i * a0;
                let x_slice = &attn.data[x_start .. x_start + in_feature];
                let af_offset = i * af0;

                for j in 0..out_feature {
                    let w_start = j * o0;
                    let w_slice = &bf16_slice[w_start .. w_start + in_feature];

                    atten_fn.data[af_offset + j] = dot_avx2_bf16(x_slice, w_slice);
                }   
            }
        }
        WeightData::Q8(q8_slice) => {
            let blocks_per_row = in_feature / 32;

            for i in 0..attn.shape[0] {
                let x_start = i * a0;
                let x_slice = &attn.data[x_start .. x_start + in_feature];
                let af_offset = i * af0;

                for j in 0..out_feature {
                    let block_start = j * blocks_per_row;
                    let block_end = block_start + blocks_per_row;
                    let w_blocks = &q8_slice[block_start .. block_end];

                    atten_fn.data[af_offset + j] = dot_avx2_q8(x_slice, w_blocks);
                }   
            }

        }
    WeightData::F32(_) => {}
    }



    Ok(())
}

pub fn out_proj_rayon(
    attn: &Tensor,
    o: &WeightTensor,
    atten_fn: &mut Tensor,
) -> Result<(), String> {
    atten_fn.update_shape(vec![attn.shape[0], attn.shape[1]]);

    assert!(
        atten_fn.data.len() >= atten_fn.shape.iter().product(),
        "attn shape wrong"
    );
    atten_fn.data.fill(0.0);

    let o0 = o.strides[0];
    let a0 = attn.strides[0];

    let o_rows = o.shape[0];
    let in_features = o.shape[1];
    //let k_len = attn.shape[1];

    let total = atten_fn.shape.iter().product();
    match &o.data {
        WeightData::BF16(bf16_slice) => {
            atten_fn.data[..total]
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, out_val)| {
                    let i = idx / o_rows;
                    let k = idx % o_rows;

                    let x_row_offset = i * a0;
                    let w_row_offset = k * o0;

                    let x_slices = &attn.data[x_row_offset..x_row_offset + o.shape[1]];
                    let w_slices = &bf16_slice[w_row_offset..w_row_offset + o.shape[1]];

                    *out_val = dot_avx2_bf16(x_slices, w_slices);
                });
        }
        WeightData::Q8(q8_slice) => {
            let blocks_per_row = in_features / 32;

            atten_fn.data[..total]
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, out_val)| {
                let i = idx / o_rows;
                let k = idx % o_rows;

                let x_row_offset = i * a0;

                let block_start = k * blocks_per_row;
                let block_end = block_start + blocks_per_row;
                let x_slices = &attn.data[x_row_offset .. x_row_offset + in_features];
                let w_blocks = &q8_slice[block_start..block_end];

                *out_val = dot_avx2_q8(x_slices, w_blocks);
            });


        }
        WeightData::F32(_) => {panic!("no f32 here");}

    }

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
    if x.shape[0] >= RAYON_THRESHOLD {
        let _ = mlp_mul_rayon(x, weight, output);
    } else {
        let _ = mlp_mul_single(x, weight, output);
    }

    Ok(())
}

pub fn mlp_mul_single(
    x: &Tensor,
    weight: &WeightTensor,
    output: &mut Tensor,
) -> Result<(), String> {
    //print!("{:?} : {:?}", x.shape, weight.shape);
    output.update_shape(vec![x.shape[0], weight.shape[0]]);
    let s1 = x.strides[0];
    let s2 = x.strides[1];
    let w1 = weight.strides[0];
    let w2 = weight.strides[1];
    let o1 = output.strides[0];
    let o2 = output.strides[1];
    //let weight_rows = weight.shape[0];
    let weight_col = weight.shape[1];

    match &weight.data {
        WeightData::BF16(bf16_w) => {
            for i in 0..x.shape[0] {
                for k in 0..weight.shape[0] {
                    let x_row_offset = i * s1;
                    let w_row_offset = k * w1;

                    let x_slice = &x.data[x_row_offset..x_row_offset + weight_col * s2];
                    let w_slices = &bf16_w[w_row_offset..w_row_offset + weight_col * w2];

                    let sum = dot_avx2_bf16(x_slice, w_slices);
                    output.data[i * o1 + k * o2] = sum;
                }
            }
        }
        WeightData::Q8(q8_w) => {
            let blocks_per_row = weight_col / 32;

            for i in 0..x.shape[0] {
                for k in 0..weight.shape[0] {
                    let x_row_offset = i * s1;

                    let block_start = k * blocks_per_row;
                    let block_end = block_start + blocks_per_row;

                    let x_slice = &x.data[x_row_offset..x_row_offset + weight_col * s2];
                    let w_slices = &q8_w[block_start..block_end];

                    let sum = dot_avx2_q8(x_slice, w_slices);
                    output.data[i * o1 + k * o2] = sum;

                }
            }
        }
        WeightData::F32(_) => {panic!("no f32 here");}
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

    let total: usize = output.shape.iter().product();

        match &weight.data {
        WeightData::BF16(bf16_w) => {
                output.data[..total]
            .par_iter_mut()
            .with_min_len(5)
            .enumerate()
            .for_each(|(idx, out_val)| {
                let i = idx / weight_rows;
                let k = idx % weight_rows;

                let x_row_offset = i * s1;
                let w_row_offset = k * w1;

                let x_slice = &x.data[x_row_offset..x_row_offset + weight_col * s2];
                let w_slices = &bf16_w[w_row_offset..w_row_offset + weight_col * w2];

                let sum = dot_avx2_bf16(x_slice, w_slices);

                *out_val = sum;
            });
        }
        WeightData::Q8(q8_w) => {
            let blocks_per_row = weight_col / 32;

                output.data[..total]
                    .par_iter_mut()
                    .with_min_len(5)
                    .enumerate()
                    .for_each(|(idx, out_val)| {

                    let i = idx / weight_rows;
                    let k = idx % weight_rows;

                    let x_row_offset = i * s1;

                    let block_start = k * blocks_per_row;
                    let block_end = block_start + blocks_per_row;

                    let x_slice = &x.data[x_row_offset..x_row_offset + weight_col * s2];
                    let w_slices = &q8_w[block_start..block_end];

                    let sum = dot_avx2_q8(x_slice, w_slices);
                    *out_val = sum;

                });
            
        }
        WeightData::F32(_) => {panic!("no f32 here");}
        }



    Ok(())
}

pub fn dot_avx2_bf16(x: &[f32], w: &[u16]) -> f32 {
    let mut sum;
    //let prefetch_dis = 32;
    unsafe {
        let mut sum_vec = _mm256_setzero_ps();
        let mut sum_vec_2 = _mm256_setzero_ps();
        let mut sum_vec_3 = _mm256_setzero_ps();
        let mut sum_vec_4 = _mm256_setzero_ps();

        let mut x_chunks = x.chunks_exact(32);
        let mut w_chunks = w.chunks_exact(32);

        for (x_chunk, w_chunk) in x_chunks.by_ref().zip(w_chunks.by_ref()) {
            let x_ptr = x_chunk.as_ptr();
            let x_vec = _mm256_loadu_ps(x_ptr);

            let x_ptr_2 = x_ptr.add(8);
            let x_vec_2 = _mm256_loadu_ps(x_ptr_2);

            let x_ptr_3 = x_ptr.add(16);
            let x_vec_3 = _mm256_loadu_ps(x_ptr_3);

            let x_ptr_4 = x_ptr.add(24);
            let x_vec_4 = _mm256_loadu_ps(x_ptr_4);


            let w_ptr = w_chunk.as_ptr();
            let w_128 = _mm_loadu_si128(w_ptr as *const __m128i);
            let w_256_int = _mm256_cvtepu16_epi32(w_128);

            let w_ptr_2 = w_ptr.add(8);
            let w_128_2 = _mm_loadu_si128(w_ptr_2 as *const __m128i);
            let w_256_int_2 = _mm256_cvtepu16_epi32(w_128_2);

            let w_ptr_3 = w_ptr.add(16);
            let w_128_3 = _mm_loadu_si128(w_ptr_3 as *const __m128i);
            let w_256_int_3 = _mm256_cvtepu16_epi32(w_128_3);

            let w_ptr_4 = w_ptr.add(24);
            let w_128_4 = _mm_loadu_si128(w_ptr_4 as *const __m128i);
            let w_256_int_4 = _mm256_cvtepu16_epi32(w_128_4);

            let w_256_shifted = _mm256_slli_epi32(w_256_int, 16);
            let w_256_shifted_2 = _mm256_slli_epi32(w_256_int_2, 16);
            let w_256_shifted_3 = _mm256_slli_epi32(w_256_int_3, 16);
            let w_256_shifted_4 = _mm256_slli_epi32(w_256_int_4, 16);
            
            let w_vec = _mm256_castsi256_ps(w_256_shifted);
            let w_vec_2 = _mm256_castsi256_ps(w_256_shifted_2);
            let w_vec_3 = _mm256_castsi256_ps(w_256_shifted_3);
            let w_vec_4 = _mm256_castsi256_ps(w_256_shifted_4);




            //here do prefetch
            //_mm_prefetch::<_MM_HINT_T0>(x_ptr.add(prefetch_dis) as *const i8);

            //_mm_prefetch::<_MM_HINT_NTA>(w_ptr.add(prefetch_dis) as *const i8);

            sum_vec = _mm256_fmadd_ps(x_vec, w_vec, sum_vec);
            sum_vec_2 = _mm256_fmadd_ps(x_vec_2, w_vec_2, sum_vec_2);
            sum_vec_3 = _mm256_fmadd_ps(x_vec_3, w_vec_3, sum_vec_3);
            sum_vec_4 = _mm256_fmadd_ps(x_vec_4, w_vec_4, sum_vec_4);
        }

        sum_vec_3 = _mm256_add_ps(sum_vec_3, sum_vec_4);
        sum_vec = _mm256_add_ps(sum_vec_2, sum_vec);
        sum_vec = _mm256_add_ps(sum_vec_3, sum_vec);

        let low_128 = _mm256_castps256_ps128(sum_vec);
        let high_128 = _mm256_extractf128_ps(sum_vec,1);

        let mut sum_128 = _mm_add_ps(low_128, high_128);
        let shuf_128 = _mm_shuffle_ps(sum_128, sum_128, 0b01_00_11_10);
        sum_128 = _mm_add_ps(sum_128,shuf_128);

        let shuf_128 = _mm_shuffle_ps(sum_128, sum_128, 0b00_01_00_01);
        sum_128 = _mm_add_ps(sum_128,shuf_128);

        sum = _mm_cvtss_f32(sum_128);

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

pub fn dot_avx2_q8(x: &[f32], w: &[BlockQ8_0]) -> f32 {
    unsafe{
        let mut acc = _mm256_setzero_ps();

        for i in 0..w.len(){
            let block = &w[i];
            let x_ptr = x.as_ptr().add(i*32);

            let v_scale = _mm256_set1_ps(block.d);

            let q_ptr = block.qs.as_ptr();

            let v_x0 = _mm256_loadu_ps(x_ptr);
            let q_chunk0 = _mm_loadl_epi64(q_ptr as *const __m128i);
            let v_q_i32_0 = _mm256_cvtepi8_epi32(q_chunk0);
            let v_q_f32_0 = _mm256_cvtepi32_ps(v_q_i32_0);
            let v_w0 = _mm256_mul_ps(v_q_f32_0, v_scale);
            acc = _mm256_fmadd_ps(v_x0, v_w0,acc);

            let v_x1 = _mm256_loadu_ps(x_ptr.add(8));
            let q_chunk1 = _mm_loadl_epi64(q_ptr.add(8) as *const __m128i);
            let v_q_i32_1 = _mm256_cvtepi8_epi32(q_chunk1);
            let v_q_f32_1 = _mm256_cvtepi32_ps(v_q_i32_1);
            let v_w1 = _mm256_mul_ps(v_q_f32_1, v_scale);
            acc = _mm256_fmadd_ps(v_x1, v_w1,acc);

            let v_x2 = _mm256_loadu_ps(x_ptr.add(16));
            let q_chunk2 = _mm_loadl_epi64(q_ptr.add(16) as *const __m128i);
            let v_q_i32_2 = _mm256_cvtepi8_epi32(q_chunk2);
            let v_q_f32_2 = _mm256_cvtepi32_ps(v_q_i32_2);
            let v_w2 = _mm256_mul_ps(v_q_f32_2, v_scale);
            acc = _mm256_fmadd_ps(v_x2, v_w2,acc);

            let v_x3 = _mm256_loadu_ps(x_ptr.add(24));
            let q_chunk3 = _mm_loadl_epi64(q_ptr.add(24) as *const __m128i);
            let v_q_i32_3 = _mm256_cvtepi8_epi32(q_chunk3);
            let v_q_f32_3 = _mm256_cvtepi32_ps(v_q_i32_3);
            let v_w3 = _mm256_mul_ps(v_q_f32_3, v_scale);
            acc = _mm256_fmadd_ps(v_x3, v_w3,acc);
        }

        let low_128 = _mm256_castps256_ps128(acc);
        let high_128 = _mm256_extractf128_ps(acc,1);
        let mut sum_128 = _mm_add_ps(low_128,high_128);

        let shuf_128 = _mm_shuffle_ps(sum_128, sum_128,0b01_00_11_10);
        sum_128 = _mm_add_ps(sum_128,shuf_128);

        let shuf_128_2 = _mm_shuffle_ps(sum_128, sum_128,0b00_01_00_01);
        sum_128 = _mm_add_ps(sum_128,shuf_128_2);

        _mm_cvtss_f32(sum_128)

    }

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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, BlockQ8_0};
    use std::vec;

    // ==========================================
    // 🛠️ 物理级探针与 Mock 锻造工具
    // ==========================================

    macro_rules! assert_f32_eq {
        ($a:expr, $b:expr) => {
            assert!(
                ($a - $b).abs() < 1e-2,
                "精度撕裂! 物理内存值左侧: {}, 右侧: {}",
                $a,
                $b
            );
        };
    }

    /// 模拟内存加载：将 f32 强行截断为 BF16
    fn mock_f32_to_bf16(val: f32) -> u16 {
        (val.to_bits() >> 16) as u16
    }

    /// 模拟铸造厂：将 f32 数组就地压缩为 Q8_0 物理块
    /// 注意：为了测试 Q8 引擎，输入数据的长度必须是 32 的倍数！
    fn mock_f32_to_q8_blocks(vals: &[f32]) -> Vec<BlockQ8_0> {
        assert!(vals.len() % 32 == 0, "物理错位：Q8 测试数据必须对齐 32 边界");
        let mut blocks = Vec::new();
        
        for chunk in vals.chunks(32) {
            let mut max_abs = 0.0f32;
            for &v in chunk {
                if v.abs() > max_abs { max_abs = v.abs(); }
            }
            
            let d = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            let mut qs = [0i8; 32];
            for i in 0..32 {
                qs[i] = (chunk[i] / d).round() as i8;
            }
            // 注意：这里需要确保 tensor.rs 里的 BlockQ8_0 的字段是 public 的，
            // 或者你可以在 tensor.rs 里给它加个 #[cfg(test)] 的新建函数
            blocks.push(BlockQ8_0 { d, qs });
        }
        blocks
    }

    // ==========================================
    // 🧪 引擎 A 测试：原生 BF16 管线
    // ==========================================

    #[test]
    fn test_token_embedding_bf16() {
        let token_ids = vec![1]; 
        let raw_f32 = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let w_data = raw_f32.iter().map(|&x| mock_f32_to_bf16(x)).collect::<Vec<u16>>();
        let weight = WeightTensor {
            data: WeightData::BF16(&w_data),
            shape: vec![2, 3], // 2个token，hidden_dim=3
            strides: vec![3, 1],
        };

        let mut out = Tensor::new(vec![], vec![0]);
        token_embedding(&token_ids, &weight, &mut out).unwrap();
        assert_f32_eq!(out.data[0], 0.4);
        assert_f32_eq!(out.data[2], 0.6);
    }

    #[test]
    fn test_rmsnorm_bf16() {
        let x = Tensor { data: vec![3.0, 4.0], shape: vec![2], strides: vec![1] };
        let w_data = vec![1.0, 2.0].iter().map(|&x| mock_f32_to_bf16(x)).collect::<Vec<u16>>();
        let weight = WeightTensor {
            data: WeightData::BF16(&w_data),
            shape: vec![2],
            strides: vec![1],
        };
        // mean_sq = 12.5, rrms ≈ 0.28284
        let out = rmsnorm(&x, &weight, 2, 1e-5).unwrap();
        assert_f32_eq!(out.data[0], 0.848528);
        assert_f32_eq!(out.data[1], 2.262741);
    }

    #[test]
    fn test_linear_proj_bf16() {
        let x = Tensor { data: vec![1.0, 2.0], shape: vec![1, 2], strides: vec![2, 1] };
        
        let w_data = vec![0.1, 0.2, 0.3, 0.4].iter().map(|&x| mock_f32_to_bf16(x)).collect::<Vec<u16>>();
        let w = WeightTensor { data: WeightData::BF16(&w_data), shape: vec![2, 2], strides: vec![2, 1] };
        
        // Linear 的 BF16 管线要求 Bias 也是 BF16
        let b_data = vec![0.1, 0.1].iter().map(|&x| mock_f32_to_bf16(x)).collect::<Vec<u16>>();
        let b = WeightTensor { data: WeightData::BF16(&b_data), shape: vec![2], strides: vec![1] };

        let mut out = Tensor::new(vec![0.0; 2], vec![0]);
        linear_proj(&x, &w, &b, &mut out).unwrap();
        
        // out[0] = 1*0.1 + 2*0.2 + 0.1 = 0.6
        // out[1] = 1*0.3 + 2*0.4 + 0.1 = 1.2
        assert_f32_eq!(out.data[0], 0.6);
        assert_f32_eq!(out.data[1], 1.2);
    }

    // ==========================================
    // 🧪 引擎 B 测试：极速 Q8_0 管线
    // ==========================================

    #[test]
    fn test_linear_proj_q8() {
        // 为了满足 Q8_0 的 32 步进要求，我们使用 in_features = 32
        let x_data = vec![1.0; 32];
        let x = Tensor { data: x_data, shape: vec![1, 32], strides: vec![32, 1] };
        
        // 权重矩阵 2 行 32 列。第一行全 1.0，第二行全 2.0
        let mut w_raw = vec![1.0; 32];
        w_raw.extend(vec![2.0; 32]);
        let q8_blocks = mock_f32_to_q8_blocks(&w_raw); // 生成 2 个 BlockQ8_0
        let w = WeightTensor { data: WeightData::Q8(&q8_blocks), shape: vec![2, 32], strides: vec![32, 1] };
        
        // ⚡ 核心断言：Q8 管线要求 Bias 必须是纯 F32！
        let b_data = vec![0.5f32, 0.5f32];
        let b = WeightTensor { data: WeightData::F32(&b_data), shape: vec![2], strides: vec![1] };

        let mut out = Tensor::new(vec![0.0; 2], vec![0]);
        linear_proj(&x, &w, &b, &mut out).unwrap();
        
        // out[0] = 32 * (1.0 * 1.0) + 0.5 = 32.5
        // out[1] = 32 * (1.0 * 2.0) + 0.5 = 64.5
        assert_f32_eq!(out.data[0], 32.5);
        assert_f32_eq!(out.data[1], 64.5);
    }

    #[test]
    fn test_rmsnorm_q8() {
        let x = Tensor { data: vec![2.0; 32], shape: vec![32], strides: vec![1] };
        // 权重全 0.5
        let w_raw = vec![0.5; 32];
        let q8_blocks = mock_f32_to_q8_blocks(&w_raw);
        let weight = WeightTensor { data: WeightData::Q8(&q8_blocks), shape: vec![32], strides: vec![1] };
        
        let out = rmsnorm_q8(&x, &weight, 32, 1e-5).unwrap();
        // RMSNorm 会把输入 2.0 归一化。mean_sq=4.0, inv_denom ≈ 0.5. norm_x = 1.0
        // 然后乘上 weight 0.5 -> 最终结果都是 0.5
        assert_f32_eq!(out.data[0], 0.5);
        assert_f32_eq!(out.data[31], 0.5);
    }

    // ==========================================
    // 🧪 纯 F32 / 数学无状态算子测试
    // ==========================================

    #[test]
    fn test_apply_rope() {
        let mut t = Tensor { data: vec![1.0, 1.0], shape: vec![1, 1, 2], strides: vec![2, 2, 1] };
        apply_rope(&mut t, 10000.0, 0); // pos=0, cos=1, sin=0
        assert_f32_eq!(t.data[0], 1.0);
        assert_f32_eq!(t.data[1], 1.0);
    }

    #[test]
    fn test_attention_score() {
        let q = Tensor { data: vec![1.0, 2.0], shape: vec![1, 1, 2], strides: vec![2, 2, 1] };
        let k = Tensor { data: vec![2.0, 3.0], shape: vec![1, 1, 2], strides: vec![2, 2, 1] };
        let mut out = Tensor::new(vec![0.0; 10], vec![0]);

        attention_score(&q, &k, 1, 1, 0, &mut out).unwrap();
        // dot product = 2.0 + 6.0 = 8.0, *= 1.0 / sqrt(2) ≈ 5.65685
        assert_f32_eq!(out.data[0], 5.65685);
    }

    #[test]
    fn test_softmax() {
        let mut t = Tensor { data: vec![0.0, 1.0], shape: vec![1, 1, 2], strides: vec![2, 2, 1] };
        softmax(&mut t);
        assert_f32_eq!(t.data[0], 0.26894);
        assert_f32_eq!(t.data[1], 0.73105);
    }

    #[test]
    fn test_silu() {
        let mut w = Tensor { data: vec![1.0], shape: vec![1], strides: vec![1] };
        let u = Tensor { data: vec![2.0], shape: vec![1], strides: vec![1] };
        // silu(1.0) * 2.0 = 0.73105 * 2 = 1.4621
        silu(&mut w, &u);
        assert_f32_eq!(w.data[0], 1.4621);
    }
}