use serde_json::Value;

pub fn raw_to_json(raw: &[u8]) -> Result<Value, String> {
    let json_string = std::str::from_utf8(raw).expect("json cannot convert to string.");
    let structure_json: Value =
        serde_json::from_str(json_string).expect("json String cannot convert to json format.");
    Ok(structure_json)
}


//test [generate by Gemini :D]
#[cfg(test)]
mod tests {
    use super::*;

    pub fn convert_to_f32(num: [u8; 2]) -> Result<f32, String> {
    Ok(f32::from_bits((u16::from_le_bytes(num) as u32) << 16))
}

    #[test]
    fn test_raw_to_json() {
        // 模拟一小段磁盘里读出来的 JSON byte 字节流
        let raw_data = b"{\"layer_count\": 24, \"model_type\": \"qwen\"}";

        let json_val = raw_to_json(raw_data).expect("JSON 内存反序列化崩溃");

        assert_eq!(json_val["layer_count"].as_i64(), Some(24));
        assert_eq!(json_val["model_type"].as_str(), Some("qwen"));
    }

    #[test]
    fn test_convert_to_f32() {
        // 物理验证 BFloat16 到 F32 的内存强转
        // 1.0f32 的标准 IEEE 754 内存是 0x3F800000
        // 它的 BF16 截断（前16位）是 0x3F80
        // 小端序 (Little Endian) 字节排列为: [0x80, 0x3F]
        let bf16_bytes_1 = [0x80, 0x3F];
        let f32_val_1 = convert_to_f32(bf16_bytes_1).expect("BF16 -> F32 转换失败");
        assert_eq!(f32_val_1, 1.0);

        // 2.0f32 的标准 IEEE 754 内存是 0x40000000
        // BF16 截断是 0x4000，小端序为: [0x00, 0x40]
        let bf16_bytes_2 = [0x00, 0x40];
        let f32_val_2 = convert_to_f32(bf16_bytes_2).expect("BF16 -> F32 转换失败");
        assert_eq!(f32_val_2, 2.0);
    }
}
