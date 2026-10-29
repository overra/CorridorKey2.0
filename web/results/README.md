# Benchmark outputs

The raw outputs behind the numbers in [../README.md](../README.md). All
browser runs: RTX 3090 (driver 615.71), Chrome 145 on Linux (Vulkan), ONNX
Runtime Web 1.31.0-dev.20260914 via transformers.js 4.3.1, written by
`node bench.mjs --out`. "fp16" is the `fp16_fused` variant, "int8" is
`q8_fused`.

In each bench JSON, `parity` compares one window against the PyTorch fp32
reference (the `golden` folder named in the file), `throughput_runs_ms` holds
the three 24-frame runs (`throughput` is the best one), and `profile` is a
slower GPU-synchronized pass that splits time by graph.

| File | What it is |
|---|---|
| `896x512_{fp16,int8}_{real,synthetic}.json` | Main results table: accuracy against the true-fp32 references (TF32 disabled), plus two of the four speed sessions. |
| `896x512_*_tf32ref.json` | The other two speed sessions per variant, from an earlier run whose references were computed with TF32 on; their `parity` numbers differ slightly from the table for that reason. |
| `896x512_fp16_no_overlap.json` | Non-overlapping windows (`--stride 4`). |
| `640x384_*.json`, `1280x768_*.json` | Other presets (synthetic test). |
| `896x512_{fp16,int8}_clip.json` | 14-frame clip against the CUDA runtime's own temporal path (`--seq golden_seq`); `sequence` holds per-frame errors. |
| `896x512_{fp16,int8}_clip_hint.json` | The same with a first-frame hint matte (`--seq golden_seq_hint`). |
| `gpu_mem_*.txt` | `nvidia-smi` memory.used samples (MiB, every 0.5 s) during a benchmark run; the first sample is the idle baseline. |
| `cuda_reference_report.json` | `onnx_export/reference.py` on the synthetic 8-frame window: per-component CUDA time (`time_bf16_s`) behind the 63 ms/frame figure. |
| `cuda_parity.txt` | `onnx_export/parity.py --variant q8_fused` on the CUDA execution provider, synthetic and real clip. The `bf16 production` lines are the bf16-autocast row of the results table. |
