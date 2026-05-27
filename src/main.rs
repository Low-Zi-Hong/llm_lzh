use memmap::Mmap;
use serde_json::Value;
use std::fs::File;

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
        token_embedding(vec![0], &embed_weight).expect("fuck");

    //undergo input layernorm
    let layernorm_weight = get_weight_matrix(
        "model.layers.0.input_layernorm.weight",
        &structure_json,
        &mmap,
        header_size,
    )
    .expect("cannot get layernorm weight.");
    let after_norm = rmsnorm(token_traits, layernorm_weight, header_size).expect("cannot run RMSnorm");

    //get all weight of QKV
    let q_weight = get_weight_matrix("model.layers.0.self_attn.q_proj.weight", &structure_json, &mmap, header_size).expect("cannot get q weight");
    let q_bias = get_weight_matrix("model.layers.0.self_attn.q_proj.bias", &structure_json, &mmap, header_size).expect("cannot get q bias");

    let q = matmul(&after_norm, &q_weight);

    print!("{:?}", q);
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

    let result_raw = &mmap[8 + header_size as usize + offset[0] as usize
        ..8 + header_size as usize + offset[1] as usize];

    let result = result_raw
        .chunks_exact(2)
        .map(|chunk| convert_to_f32([chunk[0], chunk[1]]).expect("cannot convert to f32."))
        .collect();

    Ok(Tensor {
        data: result,
        shape: shape,
        strides: vec![],
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
        strides: vec![],
    })
}

fn rmsnorm(input: Tensor, weight: Tensor, hidden_dim: usize) -> Result<Tensor, String> {
    const EPSILON: f32 = 1e-06;
    let accumulator: f32 = input.data.iter().map(|x| x * x).sum();

    let result: Vec<f32> = input
        .data
        .iter()
        .zip(weight.data.iter())
        .map(|(x, w)| (x / f32::sqrt((accumulator / hidden_dim as f32) + EPSILON)) * w)
        .collect();

    Ok(Tensor { data: result, shape: input.shape, strides: vec![] })
}

fn matmul(a:&Tensor,b:&Tensor) -> Result<Tensor,String> {

    let a_shape = a.shape.clone();
    let b_shape = b.shape.clone();
    
    let mut result = vec![0.0;a_shape[0] * b_shape[1]];

    // 3. Perform the triple loop matrix multiplication
    for i in 0..a_shape[0] {
        for j in 0..b_shape[1] {
            let mut sum = 0.0;
            for k in 0..a_shape[1] {
                // Flat index formulas for 2D array representation
                let idx_a = i * a_shape[1] + k;
                let idx_b = j * b_shape[1] + k;
                
                sum += a.data[idx_a] * b.data[idx_b];
            }
            // Store the final dot product result
            result[i * b_shape[1] + j] = sum;
        }
    }

    Ok(Tensor { data: result, shape: vec![a_shape[0] , b_shape[1]], strides: vec![] })
}