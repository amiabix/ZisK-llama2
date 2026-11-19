# ZisK-llama2

Proves inference of a Llama-2 transformer in the [ZisK](https://github.com/0xPolygonHermez/zisk) zkVM using the FP32 FloatDot precompile.

The model is Karpathy's `stories260K` from [llama2.c](https://github.com/karpathy/llama2.c), run on its original FP32 weights. Every dot product (all matmuls, RMSNorm, attention scores and outputs) is computed as an exact, ordered FP32 sum: each product and each addition rounds to nearest even, in index order. That is the same as `run.c`'s `acc += a * b` loops, bit for bit.

The guest commits three SHA-256 digests:

1. `params || tokens`: the request and the generated tokens
2. the trace: the bits of every dot-product result and every logit
3. the checkpoint: the model that was run

## Layout

- `common/`: the forward pass (`model.rs`, a port of `run.c`), the tokenizer and the request format
- `common/tests/run_c.rs`: checks that the software and FloatDot dot products both match an independent `run.c` port bit for bit
- `guest/`: the ZisK program
- `host/`: `llama2-zisk`, which runs the guest in `ziskemu` and checks its tokens and all three digests against a native run

ZisK and its FloatDot precompile come from [`amiabix/zisk@8566a014`](https://github.com/amiabix/zisk/tree/floatdot-fp32-dot).

## Run

```bash
mkdir -p models
H=https://huggingface.co/karpathy/tinyllamas/resolve/main
curl -L -o models/stories260K.bin $H/stories260K/stories260K.bin
curl -L -o models/tok512.bin      $H/stories260K/tok512.bin

cargo test --release -p llama2-common --features floatdot -- --include-ignored
cargo run --release -- --ziskemu PATH/TO/ziskemu              # software FP32
cargo run --release -- --ziskemu PATH/TO/ziskemu --floatdot   # FloatDot
cargo run --release -- --floatdot --save-input input.bin      # input for cargo-zisk
```

These need `ziskemu` and the ZisK toolchain built from the fork above.

## Emulator steps

Prompt "Once upon a" (4 tokens including BOS), 2 tokens generated greedily: " time,". That is 5 forward passes.

| | Software FP32 | FloatDot |
|---|---|---|
| Steps (this guest) | 162,428,405 | 11,071,990 |
| FloatDot calls / multiply-adds | 0 | 19,815 / 1,309,760 |

## Proving results

Measured on one NVIDIA A40 (48 GB) with `prove --gpu --minimal-memory --random-seed 0`, using the experimental FloatDot backend with a bounded setup (Blake3, uncompressed final proof).

The proofs were made with an earlier version of this guest. That version committed the request and trace digests but not the model digest, and took 165.9 M and 14.5 M steps. This repository's guest has not been proven yet.

| | Software FP32 | FloatDot, 2^14 rows | FloatDot, 2^17 rows |
|---|---|---|---|
| Instances | 1,244 | 293 | 152 |
| Main / FloatDot instances | 633 / 0 | 56 / 161 | 56 / 20 |
| Proof time | 1815.5 s | 852.9 s | **296.4 s** |
| Wall time | 1873.6 s | 867.1 s | 350.2 s |
| Peak RSS | 17.4 GB | 17.9 GB | 34.6 GB |
| Peak GPU memory | 38.5 GB | 38.4 GB | 37.4 GB |
| Proof size | 412,800 B | 412,800 B | 412,800 B |

Every proof passed `verify` and a reload check of the program key and all 64 public words. Each arm was run once.

## Limits

- `stories260K` is tiny. ZisK allows at most 2 GB of input and 512 MB of guest RAM, so a 4B-parameter model needs one proof per layer.
- Only greedy decoding is supported.
