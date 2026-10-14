# llm_lzh 🦀

A large language model inference engine written from scratch in pure Rust.

**llm_lzh** loads Hugging Face `.safetensors` weights for a Qwen-style
decoder-only transformer and runs the whole forward pass itself — every tensor
operation (matmul, RMSNorm, RoPE, grouped-query attention, SwiGLU) is
hand-written, with no PyTorch, no `candle`, no `tch`, and no BLAS. It also
carries its own BPE tokenizer, its own 8-bit weight quantizer and its own
top-p sampler, so a single `cargo run` goes from a text prompt to generated
text.

It is a learning project: the point is to understand what a modern LLM does by
building the pieces, not to compete with `llama.cpp`.

## Features

**Model loading**
- Zero-copy loading: the `.safetensors` file is `memmap`-ed and the JSON header
  is parsed with `serde_json`; weights are handed out as `WeightTensor` slices
  that point straight into the mapping (`WeightData::{F32, Bf16, Q8_0}`), so
  nothing is copied at load time.
- `config.json` drives the architecture: `num_hidden_layers`, `hidden_size`,
  `rms_norm_eps`, `num_attention_heads`, `num_key_value_heads`,
  `intermediate_size`, `vocab_size` and `rope_theta` are all read at runtime.
- Three weight dtypes are understood: `F32`, `BF16` (widened to `f32` on the
  fly with a bit shift) and the project's own `Q8_0`.

**8-bit weight quantization (`src/quantize.rs`)**
- Converts a BF16 `model.safetensors` into a `model.q8.safetensors` that the
  engine reads natively: every 2-D weight matrix is split into `Q8_0` blocks of
  32 elements (a 36-byte block: one `f32` scale plus 32 `i8` quants, quantized
  against `max_abs / 127`), while 1-D tensors — norms, biases, embeddings — are
  stored as `F32` untouched.
- The output header is rewritten with fresh `data_offsets`, padded to a 64-byte
  boundary, and `__metadata__` is dropped.
- Runs automatically at startup if `model.q8.safetensors` is missing, so the
  first `cargo run` is slower than the rest. Q8_0 weights take about 56% of the
  space of BF16.

**Hand-written operators** (all in `src/llm.rs`)
- `linear_proj` — matrix multiplication with bias, for F32/BF16/Q8_0 weights.
- `rmsnorm` — with dedicated `rmsnorm_f32`, `rmsnorm_bf16` and `rmsnorm_q8`
  kernels.
- `apply_rope` — rotary position embeddings applied to Q and K.
- `attention_score` / `softmax` / `attn_out` — scaled dot-product attention
  with grouped-query (GQA/MQA) key/value head sharing, and a numerically
  stable softmax.
- `out_proj`, `res_conn` — attention output projection and residual adds.
- `mlp_mul` + `silu` — SwiGLU: `silu(gate) * up`, then the down projection.
- `token_embedding`, `get_weight_shape`, `get_weight_matrix`.
- `dot_avx2_bf16` / `dot_avx2_q8` — dot-product kernels written with 256-bit
  AVX2 intrinsics (`_mm256_*`).

**Performance work**
- AVX2 kernels for the BF16 and Q8_0 hot paths, plus `rayon`-parallel
  `out_proj_rayon` and `mlp_mul_rayon` variants.
- A per-layer KV cache, with separate prefill and decode paths: the prompt is
  processed in one pass (all positions at once), then each new token is decoded
  one at a time against the cached keys and values.

**Tokenizer and sampling**
- `src/tokenizer.rs` implements byte-level BPE on top of a Hugging Face
  `tokenizer.json`: `added_tokens`, `model.vocab`, `model.merges` and the
  GPT-2 byte↔unicode table, with `encode` / `decode`.
- Nucleus (top-p) sampling: logits are divided by a temperature (0.6), softmaxed,
  sorted, truncated at cumulative probability 0.8, renormalized and sampled with
  a xorshift32 PRNG seeded from the clock.
- Prompts are wrapped in the Qwen ChatML template and generation stops on token
  `151643` (`<|endoftext|>`) or `151645` (`<|im_end|>`).
- The program is interactive and multi-turn: it prints `User:` and reads a line
  from stdin, streams the answer token by token, then asks again.

**Instrumentation**
- `src/bench.rs` — a `GlobalMonitor` that accumulates wall-clock time per
  operator and prints a µs-per-token report.
- `benches/ops_benchmark.rs` — `criterion` micro-benchmarks for the core ops.
- A `dhat` heap profiler hook behind a feature flag.

## Project structure

```
llm_lzh/
├── Cargo.toml                     crate metadata, deps, criterion bench target, feature flags
├── Cargo.lock                     pinned dependency versions
├── .cargo/config.toml             sets rustflags = ["-C", "target-cpu=native"]
├── .github/workflows/rust.yml     CI: cargo build + cargo test on ubuntu-latest
├── benches/
│   └── ops_benchmark.rs           criterion benchmarks for rmsnorm/rope/attention/linear_proj/mlp
├── src/
│   ├── main.rs                    the engine: config + weight loading, the big loop, sampling, REPL
│   ├── lib.rs                     library root (llm, load, tensor) so benches and tests can use the code
│   ├── llm.rs                     every operator: rmsnorm, linear_proj, RoPE, attention, SwiGLU, AVX2 kernels
│   ├── Tensor.rs                  Tensor / WeightTensor / BlockQ8_0, bf16→f32, header byte casting, strides
│   ├── load.rs                    safetensors header bytes → serde_json::Value
│   ├── quantize.rs                BF16 safetensors → Q8_0 safetensors converter
│   ├── tokenizer.rs               byte-level BPE: vocab, merges, bytes_to_unicode, encode/decode
│   └── bench.rs                   per-operator timing monitor (behind the `bench` feature)
└── response.txt                   stray debug dump of generated token IDs (not used by the code)
```

Roughly 2,800 lines of Rust. The binary (`main.rs`) declares its own modules
(`mod llm; mod tensor; ...`) and the library (`lib.rs`) exposes `llm`, `load`
and `tensor`; the benchmark target uses the library.

## Requirements

- **An x86_64 machine.** `llm.rs` uses `core::arch::x86_64` AVX2 intrinsics
  unconditionally, so the crate does not compile for `aarch64` (Apple silicon
  included) — the build fails with `unresolved import core::arch::x86_64`.
- A Rust toolchain that supports **edition 2024**, i.e. Rust 1.85 or newer
  (built and tested here with 1.99.0).
- **An AVX2-capable CPU**, because `.cargo/config.toml` compiles with
  `-C target-cpu=native` and there is no runtime CPU feature detection. On an
  older x86_64 chip the binary may hit an illegal instruction instead of
  falling back to scalar code.
- A Qwen-like model in BF16 (see below).

**Apple silicon: the AVX2 path does not port.** There is no AVX2 on arm64, so
the x86_64 binary has to run under Rosetta 2, which translates the AVX2
intrinsics to NEON. That works — measured on an M5 Pro with
Qwen2.5-0.5B-Instruct (Q8_0), it prefills 9 tokens in 194 ms and then decodes
at a median of **88 ms/token (min 49 ms, ~11 tokens/s)**, rising to ~120 ms/token
once the context is ~1.6k tokens long. The number grows with context because
`attention_score` and `softmax` are scalar and O(context) per token — at 1.6k
tokens they account for 33 ms of the 87 ms per-token total, more than the
AVX2-accelerated projections combined.

## Model files

The repository ships no weights. Put these three files in the project root
before running:

| File | Purpose |
| --- | --- |
| `config.json` | model architecture (see the key list above) |
| `model.safetensors` | the weights — **must be BF16**, the Q8_0 converter assumes 2 bytes per element |
| `tokenizer.json` | Hugging Face tokenizer (vocab, merges, added tokens) |
| `model.q8.safetensors` | *generated automatically* on first run if absent; don't commit it |

`.gitignore` excludes `*.safetensors` and `*.json`, so these files are
deliberately never tracked — clone the repo, then drop the model in yourself.

The tensor names the engine looks up are the standard Qwen/LLaMA-style ones:
`model.embed_tokens.weight`,
`model.layers.{i}.input_layernorm.weight`,
`model.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}`,
`model.layers.{i}.self_attn.o_proj.weight`,
`model.layers.{i}.post_attention_layernorm.weight`,
`model.layers.{i}.mlp.{gate,up,down}_proj.weight` and `model.norm.weight`.
Logits are computed against the **tied** embedding matrix, not a separate
`lm_head`.

Config and tokenizer paths are relative (`./config.json`, `./model.safetensors`,
`./model.q8.safetensors`, `./tokenizer.json`), so run from the project root.

## Build and run

```sh
cargo build --release
cargo run --release
```

What happens:

1. `Hello, world!` is printed, then `tokenizer.json` and `config.json` are read.
2. If `model.q8.safetensors` is missing, the BF16 weights are quantized to Q8_0
   first — this takes a while and writes a second large file next to the model.
3. The usable weights are mapped and the KV cache is allocated
   (`MAX_SEQ_LEN` = 2048 positions per layer).
4. The prompt is read from stdin, encoded, prefilled through all layers, and
   generation streams to stdout until an end token is produced. Then the
   program asks for the next turn.

## Feature flags

`bench` and `dhat_heap` are off by default; `default = []`.

```sh
cargo run --release --features bench       # per-operator µs/token report + per-token timings
cargo run --release --features dhat_heap   # dhat heap profile (stops after 20 positions)
```

With `bench` the program skips the interactive prompt and instead runs a fixed
token sequence (`vec![104022, 393, 284, 43240, 549, 18137, 99, 244, 60726]`) so
runs are repeatable, and prints a timing breakdown for each stage of the
network.

## Tests

```sh
cargo test
```

13 unit tests ship with the crate:

- `Tensor.rs` — stride/shape math, tensor allocation.
- `llm.rs` — BF16 token embedding, BF16 and Q8_0 RMSNorm, BF16 and Q8_0 linear
  projection, RoPE, attention scores, softmax, SiLU.
- `load.rs` — safetensors JSON header parsing, BF16 → F32 bit conversion.

They cover the x86_64 kernels, so they need an x86_64 host to run (see
*Requirements*). On an Apple-silicon machine the test binaries compile for
x86_64 but cannot execute without Rosetta.

There is also a GitHub Actions workflow (`.github/workflows/rust.yml`) that
builds and runs the tests on `ubuntu-latest`. **It currently fails at the build
step** — see the next section for why.

## Benchmarks

```sh
cargo bench
```

`benches/ops_benchmark.rs` uses `criterion` to measure the core operators
(RMSNorm, RoPE, attention, projections, MLP, residual) on mock 896-dimension
tensors.

## How inference works

1. **Tokenize** — the prompt is wrapped in
   `<|im_start|>system ... <|im_start|>user ... <|im_end|> <|im_start|>assistant`
   and encoded to token IDs with the BPE tokenizer.
2. **Embed** — `token_embedding` gathers the rows of `model.embed_tokens.weight`
   for those IDs.
3. **Per layer** (`num_hidden_layers` times):
   input RMSNorm → Q/K/V projections (with bias) → RoPE on Q and K → append K/V
   to the layer's KV cache → scaled dot-product attention with GQA head sharing
   → softmax → attention output → `o_proj` → residual add →
   post-attention RMSNorm → SwiGLU MLP (`gate`/`up`/`down`) → residual add.
   During prefill the whole prompt is processed in one pass; during decode only
   the new token goes through, reading the cached keys and values.
4. **Final norm and logits** — `model.norm.weight` RMSNorm, then a projection
   against the tied embedding matrix to produce one score per vocabulary entry.
5. **Sample** — temperature scale, stable softmax, top-p truncation at 0.8,
   renormalize, draw with the xorshift32 PRNG, decode the token to text, and
   loop until an end token or the context limit.

## Known limitations and rough edges

- **x86_64 only, AVX2 assumed at compile time.** No runtime dispatch, no
  scalar fallback, no GPU. `-C target-cpu=native` in `.cargo/config.toml` also
  means the binary is tuned for — and assumes — the machine it was built on.
- **The crate only builds on case-insensitive filesystems.** The file is
  committed as `src/Tensor.rs` while the code declares `mod tensor;` /
  `pub mod tensor;`; Rust resolves module files case-sensitively, so on Linux
  this fails with `error[E0583]: file not found for module 'tensor'`. macOS
  tolerates it (its default filesystem is case-insensitive), which is why it
  compiles locally but red on CI. Renaming the file to `src/tensor.rs` (or
  adding `#[path = "Tensor.rs"]`) fixes it.
- **Weights are re-fetched from the mapping on every layer of every token.**
  `get_weight_matrix` runs inside the layer loop; only the KV cache persists
  between tokens, so there is no dequantized/repacked weight cache.
- **Most operators are single-threaded scalar loops.** Only `out_proj` and
  `mlp_mul` have `rayon` variants; attention, softmax, RoPE and the norms do
  not.
- **The context window is fixed at compile time** (`MAX_SEQ_LEN = 2048`), and
  the position counter keeps growing across conversation turns; hitting the
  limit breaks out of the layer loop rather than sliding the window.
- **The Q8_0 converter is opinionated:** it assumes a BF16 source, requires
  every 2-D tensor's element count to be divisible by 32 (it asserts), and
  renumbers all offsets in a fresh header.
- **`response.txt`** (60 KB of repeated token IDs) is a leftover debugging dump
  that is tracked in git but not referenced anywhere.
- **The build is noisy:** 17 warnings, including unused imports and variables,
  two dead comparisons in the tokenizer's `bytes_to_unicode` (`b <= 0xff` on a
  `u8`, and `b > 174 && b <= 255`), and a duplicated `#[cfg(test)]` attribute.
- **Qwen-specific constants are hardcoded:** the ChatML template, the stop
  tokens, the temperature/top-p defaults and the vocabulary assumptions.
- **No license is declared**, so all rights are reserved by default.
- No batched inference, no sampling of multiple candidates, no streaming UI,
  no `--prompt` CLI flag (input always comes from stdin).

## Built with

- [`memmap`](https://crates.io/crates/memmap) — zero-copy weight mapping.
- [`serde_json`](https://crates.io/crates/serde_json) / [`serde`](https://crates.io/crates/serde) — safetensors and tokenizer parsing.
- [`rayon`](https://crates.io/crates/rayon) — data-parallel projections and MLP.
- [`criterion`](https://crates.io/crates/criterion) — micro-benchmarks.
- [`dhat`](https://crates.io/crates/dhat) — heap profiling.

## License

No license is declared for this repository yet. Until one is added, assume all
rights reserved.
