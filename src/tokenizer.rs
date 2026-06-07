use std::collections::HashMap;
use std::fs;

use serde_json::Value;

#[derive(Debug)]
pub struct Tokenizer {
    special_map: HashMap<String, u32>,
    encode_map: HashMap<String, u32>,
    decode_map: HashMap<u32, String>,
    byte_decoder: HashMap<char, u8>,
    buffer: Vec<u8>,
}

#[derive(Debug)]
enum Chunk<'a> {
    SpecialId(u32),
    Text(&'a str),
}

impl Tokenizer {
    pub fn new(vocab_path: &str) -> Self {
        let vocab_raw_str = fs::read_to_string(vocab_path).expect("cannot read config file.");
        let vocab_value: Value =
            serde_json::from_str(&vocab_raw_str).expect("cannot convert to str");

        let mut special: HashMap<String, u32> = HashMap::new();
        let mut decoder: HashMap<u32, String> = HashMap::new();
        if let Some(vocab) = vocab_value["added_tokens"].as_array() {
            //println!("running_sum");
            vocab.iter().for_each(|x| {
                let string = x["content"]
                    .as_str()
                    .expect("cannot get content")
                    .to_string();
                let id = x["id"].as_u64().expect("cannot get id") as u32;
                special.insert(string.clone(), id);
                decoder.insert(id, string);
            });
        }

        let mut encoder: HashMap<String, u32> = HashMap::new();

        if let Some(vocab) = vocab_value["model"]["vocab"].as_object() {
            for (word, index) in vocab.iter() {
                let id = index.as_f64().expect("cannot get index") as u32;
                encoder.insert(word.clone(), id);
                decoder.insert(id, word.clone());
            }
        } else {
            println!("cannot find vocab");
        }

        let mut bytedecoder = HashMap::new();
        let mut n = 0;
        for b in 0..=255u8 {
            if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b > 174 && b <= 255) {
                bytedecoder.insert(b as char, b);
            } else {
                let mapped_char = std::char::from_u32(256 + n).expect("no such unicode");
                bytedecoder.insert(mapped_char, b);
                n += 1;
            }
        }

        let token_map = Tokenizer {
            special_map: special,
            encode_map: encoder,
            decode_map: decoder,
            byte_decoder: bytedecoder,
            buffer: vec![],
        };
        //special.iter().for_each(|x| println!("{:?}",x));

        token_map
    }

    //pub fn encode(&self, text: String) -> Vec<u32> {
    //    let chunk_vec:Vec<Chunk> = vec![];
    //    let mut remaining_text = text;
    //    let mut last_id:usize = 0;
    //
    //    while !remaining_text.is_empty() {
    //        let
    //
    //    }
    //
    //
    //    for (special_str, special_id) in self.special_map.iter(){
    //        if let Some(start_idx ) = text.find(special_str) {
    //            let end_idx = start_idx + special_str.len();
    //            chunk_vec.push(Chunk::SpecialId( *special_id));
    //            chunk_vec.push(Chunk::Text(&text[last_id..start_idx]));
    //            last_id = end_idx + 1;
    //        }
    //    };
    //    chunk_vec.push(Chunk::Text(&text[last_id..text.len()]));
    //
    //    result
    //}

    pub fn decode(&mut self, token_id: u32) -> String {
        let mut result: String = "".to_string();

        let mut res = match self.decode_map.get(&token_id) {
            Some(s) => s,
            None => return "".to_string(),
        };

        for ch in res.chars() {
            if let Some(&byte) = self.byte_decoder.get(&ch) {
                self.buffer.push(byte);
            }
        }

        match std::str::from_utf8(&self.buffer) {
            Ok(valid_str) => {
                let output = valid_str.to_string();
                self.buffer.clear();
                output
            }
            Err(e) => {
                if e.error_len().is_none() {
                    "".to_string()
                } else {
                    let output = String::from_utf8_lossy(&self.buffer).into_owned();
                    self.buffer.clear();
                    output
                }
            }
        }
    }
}
