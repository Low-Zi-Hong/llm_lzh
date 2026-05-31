# llm_lzh 🦀

**llm_lzh** 是一个使用纯 Rust 从零开始编写的轻量级大语言模型（LLM）推理引擎。该项目不依赖于传统的深度学习框架，而是通过自定义的张量（Tensor）操作，硬核解析模型权重并完成 Transformer 的前向推理计算。

从代码实现来看，该项目对标了主流的开源大语言模型（如 Qwen 等），支持原生的 `.safetensors` 格式加载和典型的 LLM 网络结构。

## ✨ 核心特性

- **零拷贝加载 (Zero-copy Loading)**：通过 `memmap` 将 `.safetensors` 模型权重直接映射到内存，极大地降低了加载时间和内存开销。
- **纯手写算子**：无需外部 BLAS 库，手动实现了核心的神经网络算子：
  - 矩阵乘法 (Linear Projection)
  - RMSNorm (Root Mean Square Normalization)
  - RoPE (Rotary Positional Embeddings - 旋转位置编码)
  - 分组查询注意力机制 (GQA/MQA - Grouped-Query Attention)
  - SwiGLU 激活函数及其对应的 MLP 层
- **无缝对接 Safetensors**：自动解析 Safetensors 的 JSON Header（支持提取 `data_offsets` 和 `shape`），将二进制数据转换为浮点型 Tensor。
- **轻量级依赖**：仅仅依赖 `memmap`（内存映射）和 `serde_json`（JSON解析），保持了极高的极客范和极简风格。

## 📂 项目结构

```text
llm_lzh/
├── src/
│   ├── main.rs               # 主程序入口，包含 LLM 推理的完整大循环 (Big Loop)
│   ├── Tensor.rs             # 张量 (Tensor) 数据结构定义（包含 data, shape, strides）
│   └── SafetensorsLoader.rs  # Safetensors 权重加载模块解析器
├── Cargo.toml                # Rust 项目配置及依赖管理
└── .gitignore                # Git 忽略文件配置

```

## 🚀 快速开始

### 1. 环境要求

* 安装 [Rust 工具链](https://www.rust-lang.org/tools/install) (Edition 2024)。

### 2. 准备模型文件

在运行项目之前，你需要将目标模型的权重文件和配置文件放在项目根目录下：

1. **`config.json`**: 模型的结构配置文件（包含 `num_hidden_layers`, `hidden_size`, `rms_norm_eps`, `num_attention_heads`, `num_key_value_heads`, `rope_theta` 等字段）。
2. **`model.safetensors`**: 模型的权重文件，必须为 Safetensors 格式。

*注意：目前代码中硬编码了首批 prompt token (`vec![15144, 1351, 43415, 374]`) 且以遇到 `151643` 或 `151645` 结束，这意味着本项目原生针对类似 Qwen 这样的词表进行了测试。*

### 3. 编译与运行

克隆项目后，在根目录下执行：

```bash
cargo build --release
cargo run --release

```

## 🧠 技术细节

引擎的推理流程严格遵循了主流 Transformer 解码器 (Decoder-only) 的架构：

1. **Embedding**: `token_embedding` 从权重中提取 Token 对应的词向量。
2. **Transformer Blocks**: 遍历所有隐藏层，每一层依次进行：
* Input Layernorm (`rmsnorm`)
* QKV 投影 (`linear_proj`)
* 调整 Q/K/V 形状并应用 RoPE (`apply_rope`)
* 计算注意力得分并缩放 (`attention_score`)
* Softmax (`softmax`)
* 注意力输出 (`attn_out`) 与 Output 投影 (`out_proj`)
* 残差连接 (`res_conn`)
* Post Attention Layernorm
* MLP 网络 (使用 `silu` 激活函数计算 Gate 和 Up，再投影 Down)
* 第二次残差连接


3. **LM Head**: 对最后一层的输出应用最终的 RMSNorm，再与 Embedding weight 点乘获取 Logits，最终采用 Argmax (或分数比对) 选出下一个预测 Token。

## ⚠️ 已知限制与 TODO

* 目前所有计算均在 CPU 上利用纯标量执行，未进行 SIMD 优化或 GPU 算子加速，推理速度受到一定限制。
* `SafetensorsLoader.rs` 正在完善中，目前核心逻辑合并在 `main.rs` 内调用。
* Tokenizer 未被包含在内，当前需要手动输入 Token IDs 数组。
* 当前精度默认为 `f32`，尚不支持 int4 / int8 的 KV Cache 及权重网络量化。

```