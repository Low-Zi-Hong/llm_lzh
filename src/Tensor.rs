#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let strides = update_stride(&shape).expect("cannot create stride");
        Self {
            data,
            shape,
            strides,
        }
    }

    pub fn update_shape(&mut self, shape: Vec<usize>) {
        self.strides = update_stride(&shape).expect("cannot create stride");
        self.shape = shape;
    }
}

pub fn update_stride(shape: &Vec<usize>) -> Result<Vec<usize>, String> {
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

//test [generate by Gemini :D]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_stride_math() {
        // 1D 张量：步长永远是 [1]
        let shape1 = vec![5];
        let stride1 = update_stride(&shape1).unwrap();
        assert_eq!(stride1, vec![1]);

        // 2D 张量：比如 [3, 4] 的矩阵
        // stride 应该是 [4, 1]
        let shape2 = vec![3, 4];
        let stride2 = update_stride(&shape2).unwrap();
        assert_eq!(stride2, vec![4, 1]);

        // 3D 张量：比如 [2, 3, 4] 的 Attention QKV
        // stride 应该是 [3*4, 4, 1] = [12, 4, 1]
        let shape3 = vec![2, 3, 4];
        let stride3 = update_stride(&shape3).unwrap();
        assert_eq!(stride3, vec![12, 4, 1]);
    }

    #[test]
    fn test_tensor_new_allocation() {
        // 验证 Tensor 结构体的内存绑定与自动 Stride 计算
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];

        let tensor = Tensor::new(data.clone(), shape.clone());

        assert_eq!(tensor.data, data, "物理数据脱落");
        assert_eq!(tensor.shape, shape, "维度丢失");
        assert_eq!(tensor.strides, vec![3, 1], "自旋步长计算错误");
    }
}
