use memmap::Mmap;
use serde_json::Value;
use std::fs::{self, File};

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
    let config_path:String = "config.json".to_string();
    let config_raw = fs::read_to_string(config_path).expect("cannot read config file.");
    let config:Value = serde_json::from_str(&config_raw).expect("cannot parse config file");

    //model path
    let file_path = "qwen.safetensors";

    //open the file and use mmap to map to memory
    let file = File::open(file_path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };

    //print!("{:?}",(&mmap[0..8]));

    //get the first 8 bytes which represent the size of the next json structure
    let mut n_raw = [0u8; 8];
    n_raw.copy_from_slice(&mmap[0..8]);
    let header_size:usize = u64::from_le_bytes(n_raw) as usize;
    //print!("{:?}",n);

    //get the chunk of bit which represent the json and convert it to json value
    let json_raw = &mmap[8..8 + header_size];
    let structure_json: Value = raw_to_json(json_raw).expect("cannot run function raw_to_json");

    //println!("{}", structure_json["__metadata__"]["format"]);

    //get the size of the json
    let size_of_json = structure_json.as_object().map_or(0, |arr| arr.len());

    print!("{}", size_of_json);

    //iter through the json
    if let Some(map) = structure_json.as_object() {
        for (key, value) in map.iter().take(size_of_json) {
            //println!("{} is {}", key, value);
        }
    }

    let embed_weight = get_weight_matrix("model.embed_tokens.weight", &structure_json, &mmap, header_size).expect("cannot get embed token matrik");

    let token_traits: Tensor =
        token_embedding(vec![0,2,3,5], &embed_weight).expect("fuck");

    //undergo input layernorm
    let layernorm_weight = get_weight_matrix(
        "model.layers.0.input_layernorm.weight",
        &structure_json,
        &mmap,
        header_size,
    )
    .expect("cannot get layernorm weight.");
    let hidden_dim = config["hidden_size"].as_f64().expect("cannot get hidden_size") as usize;
    let epsilon = config["rms_norm_eps"].as_f64().expect("cannot get epsilon");
    let after_norm = rmsnorm(token_traits, layernorm_weight, hidden_dim,epsilon as f32).expect("cannot run RMSnorm");

    //get all weight of QKV
    let q_weight = get_weight_matrix("model.layers.0.self_attn.q_proj.weight", &structure_json, &mmap, header_size).expect("cannot get q weight");
    let q_bias = get_weight_matrix("model.layers.0.self_attn.q_proj.bias", &structure_json, &mmap, header_size).expect("cannot get q bias");

    let mut q = linear_proj(&after_norm, &q_weight, &q_bias).expect("cannot get q");

    let k_weight = get_weight_matrix("model.layers.0.self_attn.k_proj.weight", &structure_json, &mmap, header_size).expect("cannot get k weight");
    let k_bias =  get_weight_matrix("model.layers.0.self_attn.k_proj.bias", &structure_json, &mmap, header_size).expect("cannot get k weight");

    let mut k = linear_proj(&after_norm, &k_weight, &k_bias).expect("cannot get k");

    let v_weight = get_weight_matrix("model.layers.0.self_attn.v_proj.weight", &structure_json, &mmap, header_size).expect("cannot get v weight");
    let v_bias =  get_weight_matrix("model.layers.0.self_attn.v_proj.bias", &structure_json, &mmap, header_size).expect("cannot get v weight");

    let mut v = linear_proj(&after_norm, &v_weight, &v_bias).expect("cannot get v");

    let num_attention_heads = config["num_attention_heads"].as_u64().expect("cannot convert num to u64") as usize;
    //let num_hidden_layers = config["num_hidden_layers"].as_u64().expect("cannot convert num to u64") as usize;
    let num_key_value_heads = config["num_key_value_heads"].as_u64().expect("cannot convert num to u64") as usize;

    let mut ori_shape = q.shape.clone();
    q.shape = vec![ori_shape[0], num_attention_heads, ori_shape[1]/num_attention_heads];
    q.strides = update_stride( &q.shape).expect("cannot update stride");

    ori_shape = k.shape.clone();
    k.shape = vec![ori_shape[0], num_key_value_heads, ori_shape[1]/num_key_value_heads];
    k.strides = update_stride(&k.shape).expect("cannot update stride");

    ori_shape = v.shape.clone();
    v.shape =  vec![ori_shape[0], num_key_value_heads, ori_shape[1]/num_key_value_heads];
    v.strides = update_stride(&v.shape).expect("cannot update stride");

    //GO ROPE!!!!!!!!!!
    let rope_theta = config["rope_theta"].as_f64().expect("Cannot get rope theta") as f32;
    apply_rope(&mut q, rope_theta);
    apply_rope(&mut k, rope_theta);


    print!("{:?} : {:?}", q.shape,q.strides);
    print!("{:?}",q);
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

fn update_stride(shape:&Vec<usize>) -> Result<Vec<usize>,String> {
    let mut stride: Vec<usize> = shape.iter().rev().scan(1, |state, &dim|{
        let current_stride = *state;
        *state *= dim;
        Some(current_stride)
    }).collect();
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

    let mut stride: Vec<usize> = shape.iter().rev().scan(1, |state, &dim|{
        let current_stride = *state;
        *state *= dim;
        Some(current_stride)
    }).collect();
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

fn token_embedding(
    token_ids: Vec<usize>,
    weight_tensor: &Tensor
) -> Result<Tensor, String> {
    let hidden_dim = weight_tensor.shape[1];

    let result = token_ids
        .iter()
        .flat_map(|&id| {
            weight_tensor.data[(id * hidden_dim)..((id + 1) * hidden_dim)].iter().copied()
        })
        .collect();

    Ok(Tensor {
        data: result,
        shape: vec![token_ids.len(),hidden_dim],
        strides: vec![1,1 * hidden_dim],
    })
}

fn rmsnorm(input: Tensor, weight: Tensor, hidden_dim: usize, epsilon:f32) -> Result<Tensor, String> {


    let result: Vec<f32> = input
        .data
        .chunks_exact(hidden_dim)
        .flat_map(|chunk| {
            let accumulator: f32 = chunk.iter().map(|x| x * x).sum();
            let denominator = f32::sqrt((accumulator / hidden_dim as f32) + epsilon);
            chunk.iter().zip(weight.data.iter())
            .map(move |(&x, &w)| (x / denominator ) * w)
        }).collect();

    Ok(Tensor { data: result, shape: input.shape, strides: input.strides })
}

fn linear_proj(input:&Tensor,weight:&Tensor,bias:&Tensor) -> Result<Tensor,String> {

    let input_shape = input.shape.clone();
    let weight_shape = weight.shape.clone();
    
    let mut result = vec![0.0;input_shape[0] * weight_shape[0]];

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

    Ok(Tensor { data: result, shape: vec![input_shape[0] , weight_shape[0]], strides: vec![1,1*weight_shape[0]] })
}

fn apply_rope(tensor: &mut Tensor, rope_theta: f32){
    for pos in 0..tensor.shape[0] {
        for h in 0..tensor.shape[1]{
            for d in 0..tensor.shape[2]/2{
                let idx1 = pos * tensor.strides[0] + h * tensor.strides[1] + (d) * tensor.strides[2];
                let idx2 = pos * tensor.strides[0] + h * tensor.strides[1] + (d+(tensor.shape[2]/2)) * tensor.strides[2];

                let x1 = tensor.data[idx1];
                let x2 = tensor.data[idx2];
                
                let power:f32 = -2.0 * d as f32 / tensor.shape[2] as f32;
                let wd = rope_theta.powf(power);
                let theta = pos as f32  *wd;

                let x1n = x1 * f32::cos(theta) - x2 * f32::sin(theta);
                let x2n = x1 * f32::sin(theta) + x2 * f32::cos(theta);

                tensor.data[idx1] = x1n;
                tensor.data[idx2] = x2n;
            }
        }
    }
}