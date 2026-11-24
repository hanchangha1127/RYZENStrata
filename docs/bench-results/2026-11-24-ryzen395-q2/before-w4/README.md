# Strata standard benchmark: AMD Radeon(TM) 8060S Graphics

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261124T113040Z-cb1b5c61`, started 2026-11-24T11:30:40Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon(TM) 8060S Graphics (48.0 GiB dedicated)
- CPU: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S; RAM: 15.6 GiB
- Storage of the model files: ssd WD PC SN5000S SDEQTSJ-1T00-1002 (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 32.0.31041.1004; runtime: not measured
- Strata commit: `fe95763cafc3fd46fe1ca8372e8302dd782a86cf`; engine 0.1.67; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-q2_0; quantization Q2_0; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` `69820c02ec7d0b45...`; `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (RYZENStrata gfx1151 Windows, DFlash2 Q4_0, 4-row verification, Q8 KV, context 262144, SSD PLE; local source fixes for Windows build, CPU SKU detection and benchmark provenance/hash memory)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 20 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 59.0 (54.6-60.1) n=3 | 0.45 (0.44-0.47) n=3 | 2.53 (2.19-7.94) n=3 | 8.6 (5.9-9.8) n=3 | 1/3 |
| prompt-4k | 957.3 (957.1-961.5) n=3 | 56.6 (47.3-57.8) n=3 | 4.31 (4.30-4.33) n=3 | 9.09 (8.91-9.28) n=2 | 13.3 (13.1-15.1) n=3 | 3/3 |
| prompt-16k | 1104.6 (1102.5-1105.6) n=3 | 52.0 (49.1-56.8) n=3 | 14.89 (14.88-14.91) n=3 | 21.69 (19.31-24.77) n=3 | 24.7 (23.9-25.3) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

Sanity checks: **12 of 12**.

| Case | Result | Reason |
| --- | --- | --- |
| I1 | pass | the expected JSON value |
| I2 | pass | the expected JSON value |
| I3 | pass | the expected JSON value |
| I4 | pass | the exact expected text |
| R1 | pass | the exact expected text |
| R2 | pass | the exact expected text |
| R3 | pass | the exact expected text |
| R4 | pass | the expected JSON value |
| T1 | pass | the expected tool call |
| T2 | pass | the expected tool call |
| T3 | pass | the expected tool call |
| T4 | pass | the expected tool call |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 41 request(s) before this run.
