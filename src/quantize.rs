use std::io::Write;
use std::{collections::BTreeMap, fs::File};
use llm_lzh::tensor::Tensor;
use memmap::Mmap;
use serde_json::{Value, from_slice};
use std::collections::HashMap;
use serde::Deserialize;
use serde::Serialize;

use crate::load::raw_to_json;

#[derive(Deserialize,Serialize, Debug)]
pub struct TensorInfo {
    pub dtype : String,
    pub shape : Vec<usize>,
    pub data_offsets: [usize; 2],
}

use crate::tensor::BlockQ8_0;

//Block size is 32
pub fn generate_q8_file(file_path : &str)
{
    let file = File::open(file_path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };

    //get the first 8 bytes which represent the size of the next json structure
    let mut n_raw = [0u8; 8];
    n_raw.copy_from_slice(&mmap[0..8]);
    let header_size: usize = u64::from_le_bytes(n_raw) as usize;

    //get the chunk of bit which represent the json and convert it to json value
    let json_raw = &mmap[8..8 + header_size];

    //update json header
    let mut raw_dict :BTreeMap<String,Value> = from_slice(json_raw).expect("cannot transfer to HashMap");
    let mut current_offset:usize = 0;

    let mut new_dict :BTreeMap<String,TensorInfo> = BTreeMap::new();

    raw_dict.remove("__metadata__");

    for (tensor_name, value) in &raw_dict {
        let mut info : TensorInfo = serde_json::from_value(value.clone()).expect("cannot cast to struct");
        let element_count:usize = info.shape.iter().product();

        let size_in_bytes = if info.shape.len() == 2 {
            assert!(element_count % 32 == 0, " cannot align with 32 size block");
            info.dtype = String::from("Q8_0");
            (element_count / 32) * 36
        } else {
            info.dtype = String::from("F32");
            element_count * 4
        };

        //println!("Tensor: {}, type: {}, offset: {} -> {}",tensor_name,info.dtype,current_offset,current_offset + size_in_bytes);
        info.data_offsets = [current_offset,current_offset+size_in_bytes];
        current_offset += size_in_bytes;

        new_dict.insert(tensor_name.clone(), info);
    }

    let new_json = serde_json::to_string(&new_dict).expect("cannot convert to string");
    let new_json_byte = new_json.as_bytes();

    let mut out_file = File::create("model.q8.safetensors").expect("cannot create file");
    //out_file.write_all(b"Q8_0").expect("cannot write q8_0");
    
    let new_header_size = new_json_byte.len() as u64;
    let size_bytes = new_header_size.to_le_bytes();
    print!("{}",new_header_size);
    out_file.write_all(&size_bytes).expect("cannot write header_size");

    out_file.write_all(&new_json_byte).expect("cannot write the json body");

    for(tensor_name, value) in raw_dict {
        let mut info : TensorInfo = serde_json::from_value(value.clone()).expect("cannot cast to struct");
        let bf16_bytes = &mmap[8+header_size as usize +info.data_offsets[0] .. 8 + header_size as usize + info.data_offsets[1]];

        if info.shape.len() == 2 {
            let mut q8_block:Vec<BlockQ8_0> = Vec::with_capacity(bf16_bytes.len() / 64);

            for chunk in bf16_bytes.chunks_exact(64) {
                let mut f32_values = [0.0f32; 32];

                for i in 0..32 {
                    let u16_raw = u16::from_le_bytes([chunk[i * 2], chunk[i * 2 + 1]]);

                    let u32_shifted = (u16_raw as u32) << 16;
                    f32_values[i] = f32::from_bits(u32_shifted);
                }

                let mut max_val = 0.0f32;
                for &val in &f32_values {
                    let abs_val = val.abs();
                    if abs_val > max_val {
                        max_val = abs_val;
                    }
                }

                //here divide 127 due to an int can only hold up to 127 [-128,127];
                let scale = max_val / 127.0;
                let inv_scale = if scale == 0.0 { 0.0} else { 1.0 / scale};

                let mut qs = [0i8;32];
                for i in 0..32 {
                    let q_val = (f32_values[i] * inv_scale).round();
                    qs[i] = q_val.clamp(-127.0,127.0) as i8;
                }

                q8_block.push(BlockQ8_0 { d: scale, qs });
            }

            const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 36);

            let u8_slice:&[u8] = unsafe {
                std::slice::from_raw_parts(q8_block.as_ptr() as *const u8, q8_block.len() * std::mem::size_of::<BlockQ8_0>(),)
            };
            out_file.write_all(u8_slice).expect("cannot write data");
        } else {
            //1D tensor

            let mut f32_values: Vec<f32> = Vec::with_capacity(bf16_bytes.len() / 2);

            for chunk in bf16_bytes.chunks_exact(2) {
                let u16_raw = u16::from_le_bytes([chunk[0],chunk[1]]);
                let u32_shifted = (u16_raw as u32) << 16;
                f32_values.push(f32::from_bits(u32_shifted));
            }

            let u8_slice: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_values.as_ptr() as *const u8, f32_values.len() * std::mem::size_of::<f32>())
            };
            out_file.write_all(u8_slice).expect("cannot write data");

        }

    }
    //println!("{:?}",&raw_dict);   
    out_file.flush().expect("cannot flush outfile");

}