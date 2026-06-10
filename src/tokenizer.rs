use std::collections::HashMap;
use std::fs;

use rayon::range;
use serde_json::Value;

#[derive(Debug)]
pub struct Tokenizer {
    special_map: HashMap<String, u32>,
    encode_map: HashMap<String, u32>,
    rank_map: HashMap<([u8;16],[u8;16]),usize>,
    decode_map: HashMap<u32, String>,
    byte_encoder: HashMap<u8,char>,
    byte_decoder: HashMap<char, u8>,
    buffer: Vec<u8>,
}

#[derive(Debug)]
enum Chunk {
    SpecialId(u32),
    Text(std::ops::Range<usize>),
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

        let byteencoder = Self::bytes_to_unicode();

        let mut bytedecoder = HashMap::new();
        let mut n: u32 = 0;
        for b in 0..=255u8 {
            if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b > 174 && b <= 255) {
                bytedecoder.insert(b as char, b);
            } else {
                let mapped_char = std::char::from_u32(256 + n).expect("no such unicode");
                bytedecoder.insert(mapped_char, b);
                n += 1;
            }
        }

        let mut rank_map = HashMap::new();
        if let Some(rank) = vocab_value["model"]["merges"].as_array() {
            for (rank, item) in rank.iter().enumerate() {
                let merge_str = item.as_str().expect("merge entry not a string");
                let parts: Vec<&str> = merge_str.splitn(2, ' ').collect();
                if parts.len() != 2 { continue; }

                let mut left_buf  = [0u8; 16];
                let mut right_buf = [0u8; 16];
                let left_bytes  = parts[0].as_bytes();
                let right_bytes = parts[1].as_bytes();

                left_buf[..left_bytes.len().min(16)]
                    .copy_from_slice(&left_bytes[..left_bytes.len().min(16)]);
                right_buf[..right_bytes.len().min(16)]
                    .copy_from_slice(&right_bytes[..right_bytes.len().min(16)]);

                rank_map.insert((left_buf, right_buf), rank);
            }
        }

        let token_map = Tokenizer {
            special_map: special,
            encode_map: encoder,
            rank_map: rank_map,
            decode_map: decoder,
            byte_decoder: bytedecoder,
            byte_encoder:byteencoder,
            buffer: vec![],
        };
        //special.iter().for_each(|x| println!("{:?}",x));

        token_map
    }

    //[generate by gemini]
    pub fn bytes_to_unicode() -> HashMap<u8, char> {
        let mut b2u = HashMap::new();
        
        // 🚀 【硬核重构】：单次平推，绝对零错轨风险！
        let mut n = 0;
        for b in 0..=255 {
            // 判定当前字节是不是本来就可见的“老实人”
            let is_visible = (b >= b'!' && b <= b'~') 
                || (b >= 0xa1 && b <= 0xac) 
                || (b >= 0xae && b <= 0xff);

            if is_visible {
                // 老实人直接原样变相
                b2u.insert(b, b as char);
            } else {
                // 刺头（控制符、空格、换行）强行赋予后面的拉丁肉身！
                b2u.insert(b, char::from_u32(256 + n).unwrap());
                n += 1;
            }
        }
        b2u
    }


    pub fn encode(&self, text: &String) -> Vec<usize> {

        let mut encoded_text = String::new();

        for &b in text.as_bytes(){
            let mapped_char = self.byte_encoder.get(&b).expect("byte mapping fail");
            encoded_text.push(*mapped_char);
        }

        let chars: Vec<char> = encoded_text.chars().collect();
        let encoded_len = chars.len();

        let mut chunk_vec:Vec<Chunk> = vec![];
        let mut last_id:usize = 0;
        let mut cursor:usize = 0;
        let text_len = text.len();

        while cursor < encoded_len{
            let current_tail:String = chars[cursor..].iter().collect();
            let mut found_special = false;

            for (special_str, special_id) in self.special_map.iter() {
                if current_tail.starts_with(special_str){
                    if cursor > last_id {
                        chunk_vec.push(Chunk::Text(last_id..cursor));
                    }
                    chunk_vec.push(Chunk::SpecialId(*special_id));
                    cursor += special_str.chars().count();
                    last_id = cursor;
                    found_special = true;
                    break;
                }
            }

            if !found_special {
                cursor += 1;
            }

        }   
    
        if last_id < text.len() {
            chunk_vec.push(Chunk::Text(last_id..text.len()));
        }

        let mut result:Vec<usize> = vec![];

        for chunk in chunk_vec {
            match chunk {
                Chunk::SpecialId(id) => result.push(id as usize),
                Chunk::Text(range) => {
                    let sub_str: String = chars[range].iter().collect();

                    let bpe_token = self.bpe_merge_kernal(&sub_str);
                    result.extend(bpe_token);
                }
            }
        }
    
        result
    }

    fn bpe_merge_kernal(&self, text: &str) -> Vec<usize> {
        if text.is_empty() { return vec![];}

        let sub_bytes = text.as_bytes();

        //let mut parts: Vec<(usize,usize)> = (0..sub_bytes.len()).map(|i| (i,1)).collect();
        let mut parts: Vec<(usize, usize)> = vec![];
        let mut start = 0;
        for ch in text.chars() {
            let len = ch.len_utf8();
            parts.push((start, len));
            start += len;
        }

        loop {
            if parts.len() < 2 {break;}

            let mut best_rank = usize::MAX;
            let mut best_pair_idx = None;
            for i in 0..parts.len() - 1{
                let (p1_start, p1_len) = parts[i];
                let (p2_start, p2_len) = parts[i+1];

                let left_slice = &sub_bytes[p1_start..p1_start + p1_len];
                let right_slice = &sub_bytes[p2_start..p2_start + p2_len];

                let mut query_left = [0u8;16];
                let mut query_right = [0u8; 16];
                query_left[..left_slice.len().min(16)].copy_from_slice(&left_slice[..left_slice.len().min(16)]);
                query_right[..right_slice.len().min(16)].copy_from_slice(&right_slice[..right_slice.len().min(16)]);

                if let Some(&rank) = self.rank_map.get(&(query_left,query_right)) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair_idx = Some(i);
                    }
                }
            }   

            match best_pair_idx {
                None => break,
                Some(idx) => {
                    let (p1_start, p1_len) = parts[idx];
                    let (_, p2_len) = parts[idx+1];

                    parts[idx] = (p1_start,p1_len + p2_len);
                    parts.remove(idx+1);
                }
            }
        }

        let mut tokens = vec![];
        for (start, len) in parts {
            let final_word = &sub_bytes[start..start + len];
            let final_str = std::str::from_utf8(final_word).unwrap();

            if let Some(&id) = self.encode_map.get(final_str) {
                tokens.push(id as usize);
            }
        }

        tokens

    }

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
