use memmap::Mmap;
use serde_json::Value;
use std::io::{self, Write};
use std::{
    fs::{self, File},
    vec,
}; // Built into Rust, no extra crates needed

//for benchmark use
#[cfg(feature = "bench")]
use std::time::Instant;

const TEMPERATURE: f32 = 0.7;
const P: f32 = 0.9;
const MAX_SEQ_LEN:usize = 128;

mod tensor;
use tensor::{Tensor, update_stride};

mod llm;
use llm::{
    apply_rope, attention_score, attn_out, get_weight_matrix, linear_proj, mlp_mul, out_proj,
    random, res_conn, rmsnorm, silu, softmax, token_embedding,get_weight_shape,
};

mod load;
use load::raw_to_json;

//dhat :D
#[cfg(feature = "dhat_heap")]
use dhat;

#[cfg(feature = "dhat_heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat_heap")]
    let _profiler = dhat::Profiler::new_heap();
    println!("Hello, world!");

    #[cfg(feature = "bench")]
    let inference_start = Instant::now();
    #[cfg(feature = "bench")]
    let mut time_vec: Vec<std::time::Duration> = Vec::with_capacity(MAX_SEQ_LEN + 2);


    let raw_token = vec![106711, 44793, 53930, 99349, 3837, 99349, 34204, 17447, 99467, 35727, 3837, 100134, 53930, 99194, 3837];
    println!("Running llm with input token as: {:?}", raw_token);

    //process token
    let mut whole_token_list:Vec<usize> = raw_token.clone();
    let mut current_token:Vec<usize> = raw_token.clone();
    let mut current_pos:usize = 0;

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

    
    let layer_count = config["num_hidden_layers"]
        .as_f64()
        .expect("cannot get layer num") as usize;
    //loop start here

    let hidden_dim = config["hidden_size"]
        .as_f64()
        .expect("cannot get hidden_size") as usize;
    let epsilon = config["rms_norm_eps"].as_f64().expect("cannot get epsilon");

    let num_attention_heads = config["num_attention_heads"]
    .as_u64()
    .expect("cannot convert num to u64") as usize;

    //let num_hidden_layers = config["num_hidden_layers"].as_u64().expect("cannot convert num to u64") as usize;
    let num_key_value_heads = config["num_key_value_heads"]
        .as_u64()
        .expect("cannot convert num to u64") as usize;
    let kv_group = num_attention_heads / num_key_value_heads;

    let intermediate_size = config["intermediate_size"].as_u64().expect("cannot convert num to u64") as usize;

    let embed_weight_shape = get_weight_shape("model.embed_tokens.weight",&structure_json).expect("cannot get weight shape");
    let mut embed_weight:Tensor = Tensor::new(vec![0.0;embed_weight_shape.iter().product()], embed_weight_shape);

    get_weight_matrix(
        "model.embed_tokens.weight",
        &structure_json,
        &mmap,
        header_size,
        &mut embed_weight
    )
    .expect("cannot get embed token matrik");

    let head_dim = hidden_dim / num_attention_heads;
    //let kv_dim = hidden_dim / kv_group;

    //cache
    let mut k_cache:Vec<Tensor> = Vec::with_capacity(layer_count);
    let mut v_cache:Vec<Tensor> = Vec::with_capacity(layer_count);

    for _ in 0..layer_count{
        k_cache.push(Tensor::new(vec![0.0;MAX_SEQ_LEN*num_key_value_heads*head_dim], vec![MAX_SEQ_LEN,num_key_value_heads,head_dim]));
        v_cache.push(Tensor::new(vec![0.0;MAX_SEQ_LEN*num_key_value_heads*head_dim], vec![MAX_SEQ_LEN,num_key_value_heads,head_dim]));
    }

    let mut q_buf = Tensor::new(vec![0.0;MAX_SEQ_LEN * num_attention_heads * hidden_dim], vec![MAX_SEQ_LEN,num_attention_heads,head_dim]);
    let mut k_buf = Tensor::new(vec![0.0;MAX_SEQ_LEN * num_key_value_heads * hidden_dim], vec![MAX_SEQ_LEN, num_key_value_heads, head_dim]);
    let mut v_buf = Tensor::new(vec![0.0;MAX_SEQ_LEN * num_key_value_heads * hidden_dim], vec![MAX_SEQ_LEN, num_key_value_heads, head_dim]);

    let mut q_weight = Tensor::new(vec![0.0;hidden_dim * hidden_dim], vec![0]);
    let mut q_bias = Tensor::new(vec![0.0;hidden_dim], vec![0]);

    let mut k_weight = Tensor::new(vec![0.0;head_dim * head_dim], vec![0]);
    let mut k_bias = Tensor::new(vec![0.0;head_dim], vec![0]);

    let mut v_weight = Tensor::new(vec![0.0;head_dim * head_dim], vec![0]);
    let mut v_bias = Tensor::new(vec![0.0;head_dim], vec![0]);

    let mut o_proj = Tensor::new(vec![0.0;hidden_dim * hidden_dim], vec![0]);

    //all weight buffer
    let layernorm_shape = get_weight_shape("model.layers.0.input_layernorm.weight", &structure_json).expect("cannot get shape");
    let mut layernorm_weight = Tensor::new(vec![0.0;layernorm_shape.iter().product()], layernorm_shape);

    let mut post_attn_layernorm  = Tensor::new(vec![0.0;head_dim], vec![0]);
    let mut mlp_gate = Tensor::new(vec![0.0;intermediate_size * hidden_dim], vec![0]);
    let mut mlp_up =Tensor::new(vec![0.0;intermediate_size * hidden_dim], vec![0]);
    let mut mlp_down = Tensor::new(vec![0.0;hidden_dim * intermediate_size], vec![0]);

    #[cfg(feature = "bench")]
    println!("⚡ [BENCH] 当前 Token 准备耗时: {:?}", inference_start.elapsed());
    #[cfg(feature = "bench")]
    time_vec.push(inference_start.elapsed()  - time_vec.iter().sum());
    
    
    //big loop :D
    loop {
        let mut x: Tensor = token_embedding(&current_token, &embed_weight).expect("fuck");
        //println!("Rust Token 0 Embedding 头 5 个值: {:?}", &x.data[0..5]);

        let seq_length = current_token.len();
        
        for layer in 0..layer_count {

            if current_pos >= MAX_SEQ_LEN { break; }

            //update the buf shape
            q_buf.update_shape(vec![seq_length, num_attention_heads, head_dim]);
            k_buf.update_shape(vec![seq_length, num_key_value_heads, head_dim]);
            v_buf.update_shape(vec![seq_length,num_key_value_heads,head_dim]);

            //undergo input layernorm
            get_weight_matrix(
                &format!("model.layers.{}.input_layernorm.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
                &mut layernorm_weight
            )
            .expect("cannot get layernorm weight.");
            let x_process = rmsnorm(&x, &layernorm_weight, hidden_dim, epsilon as f32)
                .expect("cannot run RMSnorm");

            //get all weight of QKV
            get_weight_matrix(
                &format!("model.layers.{}.self_attn.q_proj.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
                &mut q_weight
            )
            .expect("cannot get q weight");
            
            get_weight_matrix(
                &format!("model.layers.{}.self_attn.q_proj.bias", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
                &mut q_bias,
            )
            .expect("cannot get q bias");
            //print!("{:?}",q_bias.shape);

            linear_proj(&x_process, &q_weight, &q_bias, &mut q_buf).expect("cannot get q");
            //println!("q is {:?}",q_buf.shape);

            get_weight_matrix(
                &format!("model.layers.{}.self_attn.k_proj.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
                &mut k_weight
            )
            .expect("cannot get k weight");
            get_weight_matrix(
                &format!("model.layers.{}.self_attn.k_proj.bias", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
                &mut k_bias
            )
            .expect("cannot get k weight");

            linear_proj(&x_process, &k_weight, &k_bias, &mut k_buf).expect("cannot get k");
            //println!("k is {:?}",&k_buf.shape);

            get_weight_matrix(
                &format!("model.layers.{}.self_attn.v_proj.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
                &mut v_weight
            )
            .expect("cannot get v weight");
            get_weight_matrix(
                &format!("model.layers.{}.self_attn.v_proj.bias", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
                &mut v_bias
            )
            .expect("cannot get v weight");

            linear_proj(&x_process, &v_weight, &v_bias, &mut v_buf).expect("cannot get v");
            //println!("v is {:?}",&v_buf.shape);

            //GO ROPE!!!!!!!!!!
            let rope_theta = config["rope_theta"]
                .as_f64()
                .expect("Cannot get rope theta") as f32;
            apply_rope(&mut q_buf, rope_theta, current_pos);
            apply_rope(&mut k_buf, rope_theta, current_pos);

            let seq_len = k_buf.shape[0];
            let chunk_size = seq_len * num_key_value_heads * head_dim;

            let start_idx = current_pos * num_key_value_heads * head_dim;
            let end_idx = start_idx + chunk_size;

            k_cache[layer].data[start_idx..end_idx].copy_from_slice(&k_buf.data[0..chunk_size]);
            v_cache[layer].data[start_idx..end_idx].copy_from_slice(&v_buf.data[0..chunk_size]);

            let valid_kv_len = current_pos + seq_length;  // 已填入的有效 KV 数量

            //Cal Score!
            let attn_start_pos = if current_pos == 0 { 
                seq_length - 1  // prefill：最后一个 token 在位置 seq_len-1
            } else { 
                current_pos     // decode：当前 token 的绝对位置
            };
            let mut s = attention_score(&q_buf, &k_cache[layer], kv_group,valid_kv_len,attn_start_pos).expect("cannot get score");   // Need new Tensor

            //softmax
            softmax(&mut s);
            //print!("{:?}", &s.data[0..s.shape[1] * s.shape[0]]);

            //cal O times value
            //print!("{:?} and {:?}", s.shape,v.shape);
            let mut attn = attn_out(&s, &v_cache[layer], kv_group).expect("cannot get attn");       // here need new Tensor
            attn.shape = vec![attn.shape[0], attn.shape[1] * attn.shape[2]];
            attn.strides = update_stride(&attn.shape).expect("cannot update stride");

            //output projection
            get_weight_matrix(
                &format!("model.layers.{}.self_attn.o_proj.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
                &mut o_proj,
            )
            .expect("cannot get o weight");

            let attn_final = out_proj(&attn, &o_proj).expect("cannot get attn final");  // here need new Tensor

            //residual connection
            //using x
            res_conn(&mut x, &attn_final);
            //print!("{:?}",&x);

            //MLP
            get_weight_matrix(
                &format!("model.layers.{}.post_attention_layernorm.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
                &mut post_attn_layernorm,
            )
            .expect("cannot get  post attention layer norm weight");
            get_weight_matrix(
                &format!("model.layers.{}.mlp.gate_proj.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
                &mut mlp_gate,
            )
            .expect("cannot get mlp gate proj");
            get_weight_matrix(
                &format!("model.layers.{}.mlp.up_proj.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
                &mut mlp_up
            )
            .expect("cannot get ml up proj");
            get_weight_matrix(
                &format!("model.layers.{}.mlp.down_proj.weight", layer).to_string(),
                &structure_json,
                &mmap,
                header_size,
                &mut mlp_down
            )
            .expect("cannot get mlp down proj");

            //post layer norm
            let post_afternorm = rmsnorm(&x, &post_attn_layernorm, hidden_dim, epsilon as f32)
                .expect("cannot perform layernorm");
            let mut gate = mlp_mul(&post_afternorm, &mlp_gate).expect("cannot mul gate");
            let up = mlp_mul(&post_afternorm, &mlp_up).expect("cannot mul up");     // here need new Tensor
            silu(&mut gate, &up);

            let ffn_x = mlp_mul(&gate, &mlp_down).expect("cannot perform mlp_mul");     // here need new Tensor
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

        let mut lm_head_weight = Tensor::new(vec![0.0;hidden_dim], vec![0]);

        get_weight_matrix("model.norm.weight", &structure_json, &mmap, header_size, &mut lm_head_weight)
            .expect("cannot get norm weight");
        let last_norm =
            rmsnorm(&last_token, &lm_head_weight, hidden_dim, epsilon as f32).expect("cannot norm");

        let mut logits = mlp_mul(&last_norm, &embed_weight).expect("lets goooooo!");

        logits.data.iter_mut().for_each(|x| *x = *x / TEMPERATURE);
        //print!("{:?}:{:?}" ,logits.shape,logits.strides);

        let max_logit = logits.data.iter().fold(-f32::INFINITY, |a, &b| a.max(b));
        let mut sum_exp = 0.0;

        logits.data.iter_mut().for_each(|x| {
            *x = f32::exp(*x - max_logit);
            sum_exp += *x;
        });

        logits.data.iter_mut().for_each(|x| *x /= sum_exp);
        let mut tuple: Vec<(usize, f32)> = logits
            .data
            .iter_mut()
            .enumerate()
            .map(|(i, val)| (i, *val))
            .collect();
        tuple.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let mut cumulative_prob = 0.0;
        let mut id: usize = 0;
        while cumulative_prob < P {
            cumulative_prob += tuple[id].1;
            id += 1;
        }

        tuple.truncate(id);

        //renormalise
        tuple
            .iter_mut()
            .for_each(|(_, val)| *val = *val / cumulative_prob);

        let random_num: f32 = random().expect("cannot generate random num") as f32;
        let mut running_sum = 0.0;
        let mut next_token_id: usize = usize::MIN;
        let mut max_score: f32 = 0.0;

        //print!("{}", random_num);

        for (i, val) in tuple.iter() {
            if random_num <= running_sum {
                next_token_id = *i;
                max_score = *val;
                break;
            }
            running_sum += *val;
        }

        if next_token_id == usize::MIN {
            next_token_id = tuple[0].0;
            max_score = tuple[0].1
        }

        //greddy search
        //let (next_token_id, max_score) = logits
        //    .data
        //    .iter()
        //    .enumerate()
        //    .max_by(|a, b| a.1.partial_cmp(b.1).expect("cannot compare"))
        //    .expect("cannot compare");
        //
        println!("next token is: {} score is {}", next_token_id, max_score);
        whole_token_list.push(next_token_id as usize);

        println!("whole token list: {:?}", &whole_token_list);

        io::stdout().flush().unwrap();

        current_pos += seq_length;
        current_token = vec![next_token_id];

        if next_token_id == 151643 || next_token_id == 151645 || current_pos >= 50{
            break;
        }

        


        //bench
        #[cfg(feature = "bench")]
        println!("⚡ [BENCH] 当前 Token 生成耗时: {:?}", inference_start.elapsed());
        #[cfg(feature = "bench")]
        time_vec.push(inference_start.elapsed() - time_vec.iter().sum());
        #[cfg(feature = "bench")]
        println!("{:?}",time_vec);

    }

    //print!("{:?}", x);
}
