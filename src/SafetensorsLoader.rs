use crate::Tensor;
use serde_json:Value;

pub struct SafttensorsLoader{
    pub mmap: memmap::Mmap,
    pub metadata: serde_json::Value,
    pub header_size: usize,
}



impl SafttensorsLoader {
    pub fn get_weight_matrix(weight_name:&str,structure_json:Value) -> Result<Tensor::Tensor,String> {

    let value = structure_json[weight_name].clone();
    let offset: Vec<usize> = value["data_offsets"]
        .as_array()
        .expect("cannot extract token offset")
        .iter()
        .map(|x| x.as_u64().expect("cannot convert num to u64.") as usize)
        .collect();
    let dtype = value["dtype"]
        .as_str()
        .expect("cannot extract dtype.");
    let shape: Vec<usize> = value["shape"]
        .as_array()
        .expect("cannot extract token shape")
        .iter()
        .map(|x| x.as_u64().expect("cannot convert num to u64.") as usize)
        .collect();

    let result = &self::mmap[8 + self::header_size as usize + offset[0] as usize
        ..8 + n as usize + offset[1] as usize];

    Ok(Tensor::Tensor { data: (), shape: vec![0,1], strides: shape })
}
}