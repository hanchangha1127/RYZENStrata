# RYZENStrata technical reference

RYZENStrata serves text with [ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO Q2_0](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF). It targets Ryzen AI Max+ 395 and Ryzen AI Max+ PRO 495 systems with an integrated `gfx1151` Radeon GPU and at least 64 GiB of installed memory. A 48 GiB or larger graphics memory budget is recommended. The 395 and 495 use Radeon 8060S and Radeon 8065S graphics respectively. [AMD GPU specifications](https://rocmdocs.amd.com/en/develop/reference/gpu-specs.html)

## Inference

- The 256K default is 262,144 tokens. K and V use the engine's `int8` cache mode by default.
- The native GPU path uses fused QSA attention with online softmax for prompt processing. Decode uses the GPU's paged attention path.
- The main weights are prepared from the publisher's Q2_0 shards. The second shard supplies the N-gram table. The prepared table stays on SSD and the engine reads its rows with direct I/O.
- The DFlash2 drafter uses Q4_0 weights. See the [original drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2).
- HIP performs model compute. The host still runs the API, tokenizer, file reads, and GPU scheduling.

The publisher reports 37.6 GB for the transformer weights and 28.8 GB for the N-gram table. These file sizes do not include the Q8 KV cache, draft model, scratch buffers, or operating-system use. A 64 GiB system with 48 GiB assigned to graphics meets the hardware target. Whether it can run the full context depends on the driver and available memory. [Model size breakdown](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF#memory-requirements)

| Component | Size on disk |
| --- | ---: |
| Q2_0 routed experts, all gate, up and down matrices | 33.974 GB (31.641 GiB) |
| Other Q2_0 tensors, including shared experts, attention and Gated DeltaNet | 3.639 GB (3.389 GiB) |
| GGUF header and metadata | 0.011 GB |
| N-gram table, separate SSD shard | 28.800 GB (26.822 GiB) |
| Default DFlash2 Q4_0 Strata pack (BF16 selector) | 0.522 GB (498 MiB) |
| Alternate Strata pack with Q4_0 selector | 0.340 GB (324 MiB) |

The Q2_0 figures come from the [first GGUF shard](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/8f752f8ae475984833f74edf84d9e43c5d1eaf98/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf) tensor directory. These weights are stored in mixed quantization. The drafter size is the packed file size.

## Local APIs

| API | Default address or endpoint |
| --- | --- |
| OpenAI base URL | `http://127.0.0.1:8080/v1` |
| Chat Completions | `POST /v1/chat/completions` |
| Responses | `POST /v1/responses` |
| Anthropic Messages | `POST /v1/messages` |

Use the API key from the current configuration if authentication is enabled. See the [setup guide](AI_SETUP.md) for the source entry points.
