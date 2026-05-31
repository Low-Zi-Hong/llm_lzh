use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use llm_lzh::{llm::{rmsnorm, apply_rope, silu, mlp_mul,linear_proj}, tensor::Tensor};

//[Generate by Gemini]
fn bench_core_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("LLM_Core_Ops");

    // ==========================================
    // 1. RMSNorm Bench (1x896 向量)
    // ==========================================
    let input_1d = Tensor::new(vec![0.5; 896], vec![1, 896]);
    let weight_1d = Tensor::new(vec![1.0; 896], vec![896]);

    group.bench_function("rmsnorm_896", |b| {
        b.iter(|| {
            // black_box 阻止编译器将计算结果当做死代码优化掉
            let out = rmsnorm(black_box(&input_1d), black_box(&weight_1d), 896, 1e-5).unwrap();
            black_box(out);
        })
    });

    // ==========================================
    // 2. SiLU 激活函数 Bench (In-place, 3584 维)
    // Qwen 中 MLP 层的中间维度通常是 896 的几倍
    // ==========================================
    let mut silu_w = Tensor::new(vec![0.5; 3584], vec![1, 3584]);
    let silu_u = Tensor::new(vec![0.5; 3584], vec![1, 3584]);

    group.bench_function("silu_3584 (in-place)", |b| {
        b.iter(|| {
            silu(black_box(&mut silu_w), black_box(&silu_u));
            black_box(&silu_w);
        })
    });

    // ==========================================
    // 3. Apply RoPE Bench (In-place)
    // 模拟 1 个 Token, 14 个 Head, 每个 Head 64 维
    // ==========================================
    let mut rope_t = Tensor::new(vec![0.5; 1 * 14 * 64], vec![1, 14, 64]);
    group.bench_function("apply_rope_1x14x64 (in-place)", |b| {
        b.iter(|| {
            apply_rope(black_box(&mut rope_t), 10000.0);
            black_box(&rope_t);
        })
    });

    // ==========================================
    // 4. MLP Mul Bench (全连接层矩阵乘法)
    // 输入 [1, 896] x 权重 [896, 3584] (目前带 vec![] 内存分配)
    // ==========================================
    let mlp_x = Tensor::new(vec![0.5; 896], vec![1, 896]);
    let mlp_w = Tensor::new(vec![0.1; 896 * 3584], vec![896, 3584]);

    group.bench_function("mlp_mul_896x3584 (with alloc)", |b| {
        b.iter(|| {
            let out = mlp_mul(black_box(&mlp_x), black_box(&mlp_w)).unwrap();
            black_box(out);
        })
    });

    group.finish();
}

// [Generate by Gemini]
fn bench_linear_proj(c: &mut Criterion) {
    // 1. 构造极其逼真的物理内存占位符
    // 模拟 Decode 阶段的 1 个 Token，特征维度 896
    let input = Tensor::new(vec![0.5; 896], vec![1, 896]);
    // 模拟 896x896 的庞大权重矩阵
    let weight = Tensor::new(vec![0.1; 896 * 896], vec![896, 896]);
    // 模拟 896 维的 bias
    let bias = Tensor::new(vec![0.0; 896], vec![896]);

    // 2. 物理插桩
    c.bench_function("linear_proj_1x896 (with Vec alloc)", |b| {
        b.iter(|| {
            // black_box 欺骗编译器：别优化我的代码，给我老老实实算完！
            let out = linear_proj(
                black_box(&input), 
                black_box(&weight), 
                black_box(&bias)
            ).expect("Bench 运行时计算图崩溃");
            
            // 确保输出也不会被编译器当成垃圾回收掉而跳过运算
            black_box(out);
        })
    });
}

criterion_group!(benches, bench_linear_proj);
criterion_group!(core_group, bench_core_ops);
criterion_main!(benches,core_group);
