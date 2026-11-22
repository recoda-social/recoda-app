#!/usr/bin/env bash
set -euo pipefail

# DGX Spark GB10 向け Qwen3.6-35B-A3B-NVFP4-Fast (vLLM + ReplaySSM) 起動スクリプト
# 注意: GPU を使う別処理（extract_slots.py 等）が終了してから起動すること

source ~/venvs/vllm-replayssm/bin/activate
export CUTE_DSL_ARCH=sm_121a
export FLASHINFER_WORKSPACE_BASE="$VIRTUAL_ENV"
export MAX_JOBS=2
export FLASHINFER_NVCC_THREADS=1

exec vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
  --served-model-name qwen3.6-35b-a3b \
  --port "${VLLM_PORT:-8000}" \
  --moe-backend flashinfer_cutlass \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.45 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prefix-caching \
  --mamba-block-size 360 \
  --prefix-match-unit 40 \
  --mamba-ssm-cache-dtype float16 \
  --mamba-backend triton \
  --use-replayssm \
  --replayssm-buffer-len 16 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 8192 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}'
