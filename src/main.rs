use memmap::Mmap;
use serde_json::Value;
use std::{
    fs::{self, File}, os::raw, vec
};
use std::io::{self, Write}; // Built into Rust, no extra crates needed


#[derive(Debug)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

pub struct LlmWeight {
    wq: Tensor, //Query Weight
    wk: Tensor, //Key Weight
    wv: Tensor, //Value Weight
}

fn main() {
    println!("Hello, world!");

    //load config.json
    let config_path: String = "config.json".to_string();
    let config_raw = fs::read_to_string(config_path).expect("cannot read config file.");
    let config: Value = serde_json::from_str(&config_raw).expect("cannot parse config file");

    //model path
    let file_path = "model.safetensors";

    //open the file and use mmap to map to memory
    let file = File::open(file_path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };

    //print!("{:?}",(&mmap[0..8]));

    //get the first 8 bytes which represent the size of the next json structure
    let mut n_raw = [0u8; 8];
    n_raw.copy_from_slice(&mmap[0..8]);
    let header_size: usize = u64::from_le_bytes(n_raw) as usize;
    //print!("{:?}",n);

    //get the chunk of bit which represent the json and convert it to json value
    let json_raw = &mmap[8..8 + header_size];
    let structure_json: Value = raw_to_json(json_raw).expect("cannot run function raw_to_json");

    //println!("{}", structure_json["__metadata__"]["format"]);

    //get the size of the json
    let size_of_json = structure_json.as_object().map_or(0, |arr| arr.len());

    //print!("{}", size_of_json);

    //iter through the json
    if let Some(map) = structure_json.as_object() {
        for (key, value) in map.iter().take(size_of_json) {
            //println!("{} is {}", key, value);
        }
    }

    let embed_weight = get_weight_matrix(
        "model.embed_tokens.weight",
        &structure_json,
        &mmap,
        header_size,
    )
    .expect("cannot get embed token matrik");

    let mut raw_token = vec![15144, 1351, 43415, 374];

    //big loop :D
    loop {
        let mut x: Tensor = token_embedding(&raw_token, &embed_weight).expect("fuck");
        //println!("Rust Token 0 Embedding 头 5 个值: {:?}", &x.data[0..5]);

        let layer_count = config["num_hidden_layers"]
            .as_f64()
            .expect("cannot get layer num") as usize;

        //loop start here

        let hidden_dim = config["hidden_size"]
            .as_f64()
            .expect("cannot get hidden_size") as usize;
        let epsilon = config["rms_norm_eps"].as_f64().expect("cannot get epsilon");

        for layer in 0..layer_count {
            //undergo input layernorm
            let layernorm_weight = get_weight_matrix(
                &format!("model.layers.{}.input_layernorm.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
            )
            .expect("cannot get layernorm weight.");
            let after_norm = rmsnorm(&x, &layernorm_weight, hidden_dim, epsilon as f32)
                .expect("cannot run RMSnorm");

            //get all weight of QKV
            let q_weight = get_weight_matrix(
                &format!("model.layers.{}.self_attn.q_proj.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
            )
            .expect("cannot get q weight");
            let q_bias = get_weight_matrix(
                &format!("model.layers.{}.self_attn.q_proj.bias", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
            )
            .expect("cannot get q bias");

            let mut q = linear_proj(&after_norm, &q_weight, &q_bias).expect("cannot get q");

            let k_weight = get_weight_matrix(
                &format!("model.layers.{}.self_attn.k_proj.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
            )
            .expect("cannot get k weight");
            let k_bias = get_weight_matrix(
                &format!("model.layers.{}.self_attn.k_proj.bias", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
            )
            .expect("cannot get k weight");

            let mut k = linear_proj(&after_norm, &k_weight, &k_bias).expect("cannot get k");

            let v_weight = get_weight_matrix(
                &format!("model.layers.{}.self_attn.v_proj.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
            )
            .expect("cannot get v weight");
            let v_bias = get_weight_matrix(
                &format!("model.layers.{}.self_attn.v_proj.bias", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
            )
            .expect("cannot get v weight");

            let mut v = linear_proj(&after_norm, &v_weight, &v_bias).expect("cannot get v");

            let num_attention_heads = config["num_attention_heads"]
                .as_u64()
                .expect("cannot convert num to u64") as usize;
            //let num_hidden_layers = config["num_hidden_layers"].as_u64().expect("cannot convert num to u64") as usize;
            let num_key_value_heads = config["num_key_value_heads"]
                .as_u64()
                .expect("cannot convert num to u64") as usize;
            let kv_group = num_attention_heads / num_key_value_heads;

            let mut ori_shape = q.shape.clone();
            q.shape = vec![
                ori_shape[0],
                num_attention_heads,
                ori_shape[1] / num_attention_heads,
            ];
            q.strides = update_stride(&q.shape).expect("cannot update stride");

            ori_shape = k.shape.clone();
            k.shape = vec![
                ori_shape[0],
                num_key_value_heads,
                ori_shape[1] / num_key_value_heads,
            ];
            k.strides = update_stride(&k.shape).expect("cannot update stride");

            ori_shape = v.shape.clone();
            v.shape = vec![
                ori_shape[0],
                num_key_value_heads,
                ori_shape[1] / num_key_value_heads,
            ];
            v.strides = update_stride(&v.shape).expect("cannot update stride");

            //GO ROPE!!!!!!!!!!
            let rope_theta = config["rope_theta"]
                .as_f64()
                .expect("Cannot get rope theta") as f32;
            apply_rope(&mut q, rope_theta);
            apply_rope(&mut k, rope_theta);

            //Cal Score!
            let mut s = attention_score(&q, &k, kv_group).expect("cannot get score");

            //softmax
            softmax(&mut s);
            //print!("{:?}", &s.data[0..s.shape[1] * s.shape[0]]);

            //cal O times value
            let mut attn = attn_out(&s, &v, kv_group).expect("cannot get attn");
            attn.shape = vec![attn.shape[0], attn.shape[1] * attn.shape[2]];
            attn.strides = update_stride(&attn.shape).expect("cannot update stride");

            //output projection
            let o_proj = get_weight_matrix(
                &format!("model.layers.{}.self_attn.o_proj.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
            )
            .expect("cannot get o weight");
            let attn_final = out_proj(&attn, &o_proj).expect("cannot get attn final");

            //residual connection
            //using x
            res_conn(&mut x, &attn_final);
            //print!("{:?}",&x);

            //MLP
            let post_attn_layernorm = get_weight_matrix(
                &format!("model.layers.{}.post_attention_layernorm.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
            )
            .expect("cannot get  post attention layer norm weight");
            let mlp_gate = get_weight_matrix(
                &format!("model.layers.{}.mlp.gate_proj.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
            )
            .expect("cannot get mlp gate proj");
            let mlp_up = get_weight_matrix(
                &format!("model.layers.{}.mlp.up_proj.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
            )
            .expect("cannot get ml up proj");
            let mlp_down = get_weight_matrix(
                &format!("model.layers.{}.mlp.down_proj.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
            )
            .expect("cannot get mlp down proj");

            //post layer norm
            let post_afternorm = rmsnorm(&x, &post_attn_layernorm, hidden_dim, epsilon as f32)
                .expect("cannot perform layernorm");
            let mut gate = mlp_mul(&post_afternorm, &mlp_gate).expect("cannot mul gate");
            let up = mlp_mul(&post_afternorm, &mlp_up).expect("cannot mul up");
            silu(&mut gate, &up);

            let ffn_x = mlp_mul(&gate, &mlp_down).expect("cannot perform mlp_mul");
            res_conn(&mut x, &ffn_x);
        }

        //last!!!!!
        let last_token_data =
            x.data[(x.shape[0] - 1) * hidden_dim..x.shape[0] * hidden_dim].to_vec();
        let last_token: Tensor = Tensor {
            data: last_token_data,
            shape: vec![1, hidden_dim],
            strides: update_stride(&vec![1, hidden_dim]).expect("cannot get stride"),
        };

        let lm_head_weight =
            get_weight_matrix("model.norm.weight", &structure_json, &mmap, header_size)
                .expect("cannot get norm weight");
        let last_norm =
            rmsnorm(&last_token, &lm_head_weight, hidden_dim, epsilon as f32).expect("cannot norm");

        let logits = mlp_mul(&last_norm, &embed_weight).expect("lets goooooo!");

        let (next_token_id, max_score) = logits
            .data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("cannot compare"))
            .expect("cannot compare");

        println!("next token is: {} score is {}", next_token_id, max_score);
        raw_token.push(next_token_id as usize);
        println!("whole token list: {:?}", raw_token);
        io::stdout().flush().unwrap(); 
        if next_token_id == 151643 || next_token_id == 151645 {break;}
    }

    //print!("{:?}", x);
}

#[inline(never)]
fn raw_to_json(raw: &[u8]) -> Result<Value, String> {
    let json_string = std::str::from_utf8(raw).expect("json cannot convert to string.");
    let structure_json: Value =
        serde_json::from_str(json_string).expect("json String cannot convert to json format.");
    Ok(structure_json)
}

fn convert_to_f32(num: [u8; 2]) -> Result<f32, String> {
    Ok(f32::from_bits((u16::from_le_bytes(num) as u32) << 16))
}

fn update_stride(shape: &Vec<usize>) -> Result<Vec<usize>, String> {
    let mut stride: Vec<usize> = shape
        .iter()
        .rev()
        .scan(1, |state, &dim| {
            let current_stride = *state;
            *state *= dim;
            Some(current_stride)
        })
        .collect();
    stride.reverse();
    Ok(stride)
}

fn get_weight_matrix(
    weight_name: &str,
    structure_json: &Value,
    mmap: &Mmap,
    header_size: usize,
) -> Result<Tensor, String> {
    let value = structure_json[weight_name].clone();
    let offset: Vec<usize> = value["data_offsets"]
        .as_array()
        .expect("cannot extract token offset")
        .iter()
        .map(|x| x.as_u64().expect("cannot convert num to u64.") as usize)
        .collect();
    //let dtype = value["dtype"].as_str().expect("cannot extract dtype.");
    let shape: Vec<usize> = value["shape"]
        .as_array()
        .expect("cannot extract token shape")
        .iter()
        .map(|x| x.as_u64().expect("cannot convert num to u64.") as usize)
        .collect();

    let mut stride: Vec<usize> = shape
        .iter()
        .rev()
        .scan(1, |state, &dim| {
            let current_stride = *state;
            *state *= dim;
            Some(current_stride)
        })
        .collect();
    stride.reverse();

    let result_raw = &mmap[8 + header_size as usize + offset[0] as usize
        ..8 + header_size as usize + offset[1] as usize];

    let result = result_raw
        .chunks_exact(2)
        .map(|chunk| convert_to_f32([chunk[0], chunk[1]]).expect("cannot convert to f32."))
        .collect();

    Ok(Tensor {
        data: result,
        shape: shape,
        strides: stride,
    })
}

fn token_embedding(token_ids: &Vec<usize>, weight_tensor: &Tensor) -> Result<Tensor, String> {
    let hidden_dim = weight_tensor.shape[1];

    let result = token_ids
        .iter()
        .flat_map(|&id| {
            weight_tensor.data[(id * hidden_dim)..((id + 1) * hidden_dim)]
                .iter()
                .copied()
        })
        .collect();

    Ok(Tensor {
        data: result,
        shape: vec![token_ids.len(), hidden_dim],
        strides: vec![1 * hidden_dim, 1],
    })
}

fn rmsnorm(
    input: &Tensor,
    weight: &Tensor,
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
                .map(move |(&x, &w)| (x / denominator) * w)
        })
        .collect();

    Ok(Tensor {
        data: result,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    })
}

fn linear_proj(input: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor, String> {
    let input_shape = input.shape.clone();
    let weight_shape = weight.shape.clone();

    let mut result = vec![0.0; input_shape[0] * weight_shape[0]];

    // 3. Perform the triple loop matrix multiplication
    for i in 0..input_shape[0] {
        for j in 0..weight_shape[0] {
            let mut sum = 0.0;
            for k in 0..input_shape[1] {
                // Flat index formulas for 2D array representation
                let idx_input = i * input_shape[1] + k;
                let idx_weight = j * weight_shape[1] + k;

                sum += input.data[idx_input] * weight.data[idx_weight];
            }
            // Store the final dot product result
            result[i * weight_shape[0] + j] = sum + bias.data[j];
        }
    }

    Ok(Tensor {
        data: result,
        shape: vec![input_shape[0], weight_shape[0]],
        strides: vec![1, 1 * weight_shape[0]],
    })
}

fn apply_rope(tensor: &mut Tensor, rope_theta: f32) {
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
                let theta = pos as f32 * wd;

                let x1n = x1 * f32::cos(theta) - x2 * f32::sin(theta);
                let x2n = x1 * f32::sin(theta) + x2 * f32::cos(theta);

                tensor.data[idx1] = x1n;
                tensor.data[idx2] = x2n;
            }
        }
    }
}

fn attention_score(q: &Tensor, k: &Tensor, kv_group: usize) -> Result<Tensor, String> {
    let mut s: Tensor = Tensor {
        data: vec![0.0; q.shape[1] * q.shape[0] * k.shape[0]],
        shape: vec![q.shape[1], q.shape[0], k.shape[0]],
        strides: vec![],
    };

    //update stride
    s.strides = update_stride(&s.shape).expect("cannot generate stride for S");
    //print!("S shape: {:?} S stride:  {:?} S data size: {:?}",&s.shape, &s.strides, &s.data.len());
    for h in 0..s.shape[0] {
        //iter through head
        for pos_q in 0..s.shape[1] {
            for pos_k in 0..s.shape[2] {
                let idx = h * s.strides[0] + pos_q * s.strides[1] + pos_k * s.strides[2] as usize;
                if pos_q >= pos_k {
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

    Ok(s)
}

fn softmax(t: &mut Tensor) {
    for h in 0..t.shape[0] {
        for pos_q in 0..t.shape[1] {
            let mut max = -f32::INFINITY;
            let start_idx = h * t.strides[0] + pos_q * t.strides[1];
            let mut idx = start_idx;
            for r in 0..t.shape[2] {
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

fn attn_out(s: &Tensor, v: &Tensor, kv_group: usize) -> Result<Tensor, String> {
    let mut o: Tensor = Tensor {
        data: vec![],
        shape: vec![v.shape[0], s.shape[0], v.shape[2]],
        strides: vec![],
    };
    o.strides = update_stride(&o.shape).expect("cannot generate stride");
    o.data = vec![0.0 as f32; o.shape[0] * o.shape[1] * o.shape[2]];

    for pos_q in 0..o.shape[0] {
        for h in 0..o.shape[1] {
            for d in 0..o.shape[2] {
                let mut sum = 0.0 as f32;
                for pos_k in 0..s.shape[2] {
                    let idx_s = h * s.strides[0] + pos_q * s.strides[1] + pos_k * s.strides[2];
                    let idx_v =
                        pos_k * v.strides[0] + (h / kv_group) * v.strides[1] + d * v.strides[2];
                    sum += s.data[idx_s] * v.data[idx_v];
                }
                let idx = pos_q * o.strides[0] + h * o.strides[1] + d * o.strides[2];
                o.data[idx] = sum;
            }
        }
    }

    Ok(o)
}

fn out_proj(attn: &Tensor, o: &Tensor) -> Result<Tensor, String> {
    let mut atten_fn: Tensor = Tensor {
        data: vec![],
        shape: vec![attn.shape[0], attn.shape[1]],
        strides: vec![],
    };
    atten_fn.strides = update_stride(&atten_fn.shape).expect("cannot get stride");
    atten_fn.data = vec![0.0 as f32; atten_fn.shape[0] * atten_fn.shape[1]];

    for i in 0..attn.shape[0] {
        for j in 0..o.shape[0] {
            let mut sum = 0.0f32;
            for k in 0..attn.shape[1] {
                sum += attn.data[i * attn.strides[0] + k] * o.data[j * o.strides[0] + k];
            }
            atten_fn.data[i * atten_fn.strides[0] + j] = sum;
        }
    }

    Ok(atten_fn)
}

fn res_conn(x: &mut Tensor, attn_final: &Tensor) {
    //print!("{:?}",x.shape);
    x.data
        .iter_mut()
        .zip(attn_final.data.iter())
        .for_each(|(a, b)| *a += *b);
}

fn mlp_mul(x: &Tensor, weight: &Tensor) -> Result<Tensor, String> {
    //print!("{:?} : {:?}", x.shape, weight.shape);
    let mut output: Tensor = Tensor {
        data: vec![],
        shape: vec![x.shape[0], weight.shape[0]],
        strides: vec![],
    };
    output.strides = update_stride(&output.shape).expect("cannot upoadte stride");
    output.data = vec![0.0; output.shape[0] * output.shape[1]];

    for i in 0..x.shape[0] {
        for k in 0..weight.shape[0] {
            let mut sum = 0.0;
            for j in 0..x.shape[1] {
                let x_val = x.data[i * x.strides[0] + j * x.strides[1]];
                let w_val = weight.data[k * weight.strides[0] + j * weight.strides[1]];

                sum += x_val * w_val;
            }

            output.data[i * output.strides[0] + k * output.strides[1]] = sum;
        }
    }

    Ok(output)
}

fn silu(weight: &mut Tensor, up: &Tensor) {
    weight
        .data
        .iter_mut()
        .zip(up.data.iter())
        .for_each(|(x, u)| *x = (*x / (1.0 + f32::exp(-*x))) * *u);
}
