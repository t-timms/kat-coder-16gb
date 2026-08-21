# Environment

This pipeline needs **three separate Python environments**. They cannot be merged,
and the reason is not tidiness.

| environment | holds | why it must be separate |
|---|---|---|
| `~/reap-cuda-env` | reap fork, transformers 5.15 | pruning needs transformers >= 5.2 to load `qwen3_5_moe` |
| `~/quant-env` | llm-compressor 0.13, compressed-tensors, transformers 5.14 | quantization needs both llm-compressor and a transformers new enough to load the model |
| `~/vllm-env` | vLLM 0.20.2 built from source for SM120, lm-eval | **transformers 4.57.1**, which cannot load `qwen3_5_moe` at all |

The vLLM environment deliberately keeps an old transformers. vLLM ships its own
model implementations, so it does not need transformers to understand the
architecture, and upgrading 4.x to 5.x risks breaking a working SM120 source build
that took considerable effort to produce. Do not "fix" it.

## Pinned versions

Two distinct environments matter and they are not the same. The published speed and
CUDA-graph findings were measured on the benchmark-era stack; the machine was later
rebuilt onto a newer one, which is what the accuracy reproduction ran on. Both are
recorded because conflating them is how a result stops being reproducible.

| component | benchmark-era | currently installed |
|---|---|---|
| `reap` | `t-timms/reap-cuda` @ `2954ba3` (`fix/qwen3-5-router-renormalization`) | same |
| `llmcompressor` | 0.13.0 | 0.13.0 |
| `compressed-tensors` (quant) | 0.18.0 | 0.18.0 |
| `transformers` (prune / quant) | 5.15.x / 5.14.1 | 5.15.1 / 5.14.1 |
| `torch` (prune / quant) | 2.13.0+cu130 | 2.13.0+cu130 / 2.13.0 |
| `vllm` | 0.20.2, source build | **0.26.0+cu131**, source build |
| `transformers` (serving) | 4.57.1 | **5.15.1** |
| `compressed-tensors` (serving) | — | 0.17.0 |
| `lm_eval` | 0.4.x | 0.4.12 |

The two bolded rows are the ones to watch: the serving environment moved forward
after the benchmark run. Reproducing **149.5 tok/s and the CUDA-graph behaviour
requires the 0.20.2 build**; the accuracy figures are insensitive to it and were
reproduced on 0.26.0. Regenerate this table with
`bash scripts/probes/capture_environment.sh`.

There is also a fourth environment, `~/swebench-env`, used by the SWE-bench harness
and `scripts/eval/read_scores.py`. It holds no model code.

## Hardware assumed

- NVIDIA RTX 5070 Ti, 16 GB, compute capability 12.0 (SM120), PCIe 4.0 x16
- 96 GB system RAM, of which WSL2 is allocated 80 GB (`memory=80GB` in `.wslconfig`)
- Roughly 250 GB of free disk for checkpoints

The 80 GB WSL allocation is deliberate. REAP holds a 65.4 GiB checkpoint entirely in
RAM during calibration, because its observer cannot read weights that accelerate has
offloaded to disk. Allocating too much is also dangerous: at 88 GB the host ran out
of commit and hard-froze, since the Linux OOM killer cannot protect Windows.

## Building the environments

```bash
# 1. pruning
python3 -m venv ~/reap-cuda-env
~/reap-cuda-env/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
# The FORK, not upstream. It carries the router renormalization fix this build
# depends on; upstream reap silently disables renormalization for this
# architecture and produces a materially different model. The pruned checkpoint
# is named reap-renorm_true-seed_42-0.50 for exactly this reason.
git clone -b fix/qwen3-5-router-renormalization \
  https://github.com/t-timms/reap-cuda ~/reap-cuda
git -C ~/reap-cuda checkout 2954ba3d364c10a41211d3f5b8957032414bc0cc
# installed with --no-deps: its vllm==0.10.0 / torch==2.7.1 pins conflict with the
# CUDA 13 build this card needs
~/reap-cuda-env/bin/pip install --no-deps -e ~/reap-cuda
~/reap-cuda-env/bin/pip install transformers>=5.2 datasets accelerate safetensors

# 2. quantization
python3 -m venv ~/quant-env
~/quant-env/bin/pip install llmcompressor==0.13.0 "transformers>=5.2" datasets

# 3. serving and evaluation: vLLM built from source with SM120 kernels
#    (a stock wheel will not do; see the SM120 notes in the README)
python3 -m venv ~/vllm-env
# ... source build of vLLM 0.20.2 ...
~/vllm-env/bin/pip install lm-eval
```

## Model weights

```bash
hf download Kwaipilot/KAT-Coder-V2.5-Dev --local-dir ~/models/KAT-Coder-V2.5-Dev
```

69.3 GB in bf16. A pre-quantized unpruned NVFP4 build also exists at
`sakamakismile/KAT-Coder-V2.5-Dev-NVFP4` (21.9 GB), useful as a control, though it
carries a disclosed accuracy warning about differing global scales on fused layers,
and it will not fit 16 GB without CPU offload, which crashes on this stack.

## Known environment traps

- `gh` is not installed inside WSL, only `gh.exe` through interop. `git push` from
  WSL hangs forever unless `credential.helper` is set to
  `!gh.exe auth git-credential`.
- `gh.exe` cannot interpret a WSL working directory as a git repository, so
  `gh repo create --source=.` fails from inside WSL. Create the repo bare and add
  the remote with git instead.
- vLLM reports `pin_memory=False` under WSL and warns that this costs performance.
- Do not inline `$(...)`, `python -c`, or shell loops into `wsl -- bash -lc` from
  PowerShell. Variables are expanded by PowerShell before bash sees them, and the
  failures are silent as often as loud. Write a script file, copy it in, strip
  carriage returns, then run it.
