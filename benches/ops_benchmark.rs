use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use llm_lzh::{
    llm::{apply_rope, linear_proj, mlp_mul, rmsnorm, silu, attention_score,res_conn, attn_out, softmax},
    tensor::Tensor,
};

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
            apply_rope(black_box(&mut rope_t), 10000.0,0);
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

    // ==========================================
    // 5. Linear Proj Bench (零分配 In-place 矩阵乘法)
    // 假设输入 [1, 896], 权重 [896, 896], 无/有 Bias
    // 用来和上面带 alloc 的 mlp_mul 形成惨烈对比！
    // ==========================================
    let lin_x = Tensor::new(vec![0.5; 896], vec![1, 896]);
    let lin_w = Tensor::new(vec![0.1; 896 * 896], vec![896, 896]);
    let lin_bias = Tensor::new(vec![0.0; 896], vec![896]); // 如果你的签名需要 bias
    let mut lin_out = Tensor::new(vec![0.0; 896], vec![1, 896]);

    group.bench_function("linear_proj_896x896 (in-place)", |b| {
        b.iter(|| {
            // 请根据你实际的 linear_proj 签名调整参数
            linear_proj(black_box(&lin_x), black_box(&lin_w), black_box(&lin_bias), black_box(&mut lin_out)).unwrap();
            black_box(&lin_out);
        })
    });

    // ==========================================
    // 6. Attention Score Bench (Q @ K.T)
    // 模拟 1 个 Q 去找 1024 个历史 K 算分数 (模拟 Decode 中期)
    // ==========================================
    // 假设 1 个 head，head_dim = 64
    let q_attn = Tensor::new(vec![0.5; 64], vec![1, 1, 64]);
    // 假装这是 k_cache 里的一层，已经存了 1024 个 token
    let k_attn = Tensor::new(vec![0.5; 1024 * 1 * 64], vec![1024, 1, 64]); 
    
    group.bench_function("attention_score_1x1024 (in-place)", |b| {
        b.iter(|| {
            // 假设你的参数是 (q, k, seq_len/pos, heads, head_dim) 等，请对齐你的真实签名
            // 注意：如果内部有 vec![] 分配，这个耗时会明显飙升
            let score = attention_score(black_box(&q_attn), black_box(&k_attn), black_box(1), black_box(1024), black_box(1024)).unwrap();
            black_box(score);
        })
    });

    // ==========================================
    // 7. Residual Connection Bench (残差连接)
    // 纯粹的内存带宽极限测试 (1x896 向量加法)
    // ==========================================
    let mut res_x = Tensor::new(vec![0.5; 896], vec![1, 896]);
    let res_y = Tensor::new(vec![0.1; 896], vec![1, 896]);

    group.bench_function("res_conn_896 (in-place)", |b| {
        b.iter(|| {
            // 假设你的 res_conn 是把 y 加到 x 上
            res_conn(black_box(&mut res_x), black_box(&res_y));
            black_box(&res_x);
        })
    });

    // ==========================================
    // 8. Softmax Bench (In-place)
    // 物理场景：当前生成 1 个 Token，有 14 个 Q 头，历史长度为 1024
    // 必须是 In-place 原位修改，否则分配内存的开销会比算 exp() 还大！
    // ==========================================
    let mut score_buf = Tensor::new(vec![0.5; 14 * 1024], vec![1, 14, 1024]);
    
    group.bench_function("softmax_1x14x1024 (in-place)", |b| {
        b.iter(|| {
            // ⚠️ 注意：请根据你实际的 softmax 函数签名调整
            // 理论上它只需要一个 &mut Tensor 即可
            softmax(black_box(&mut score_buf));
            black_box(&score_buf);
        })
    });

    // ==========================================
    // 9. Attn Out Bench (Score @ V)
    // 物理场景：拿 [14, 1024] 的概率矩阵，去乘 [1024, 14, 64] 的 V Cache
    // 这是整个 Attention 里最大的“访存刺客”！
    // ==========================================
    // 假装这是算完 Softmax 的概率矩阵
    let prob_buf = Tensor::new(vec![0.001; 14 * 1024], vec![1, 14, 1024]);
    // 假装这是已经装了 1024 个历史 Token 的当前层 V Cache 塔 (假设 V 头也是 14)
    // 如果你的模型是 GQA (比如 V 只有 2 个头)，请自行把这里的 14 改成 2！
    let v_cache_layer = Tensor::new(vec![0.1; 1024 * 14 * 64], vec![1024, 14, 64]);
    // 准备好接收结果的坑位
    let mut attn_out_buf = Tensor::new(vec![0.0; 14 * 64], vec![1, 14, 64]);

    group.bench_function("attn_out_1x14x64 (in-place)", |b| {
        b.iter(|| {
            // ⚠️ 注意：请根据你实际的 attn_out 函数签名调整
            // 比如有些实现需要传入 heads 数量或者 current_pos 边界
            attn_out_buf = attn_out(
                black_box(&prob_buf), 
                black_box(&v_cache_layer), 
                2 // 👈 必须是零分配传参！
                // , black_box(1024) // 如果你需要传 current_pos 的话
            ).expect("fuck");
            black_box(&attn_out_buf);
        })
    });

    group.finish();
}


criterion_group!(core_group, bench_core_ops);
criterion_main!(core_group);
