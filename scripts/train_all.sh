#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

BASE_CONFIG=${BASE_CONFIG:-configs/baseline.yaml}
DATA_CONFIG_DIR=${DATA_CONFIG_DIR:-configs/datasets/main}
MODEL_CONFIG_DIR=${MODEL_CONFIG_DIR:-configs/models/main}
LOSS_CONFIG_DIR=${LOSS_CONFIG_DIR:-configs/losses/main}
NUM_GPUS=${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}
LOG_DIR=${LOG_DIR:-logs/train}

if (( NUM_GPUS < 1 )); then
  echo "No CUDA device is available."
  exit 1
fi

mapfile -t DATA_CONFIGS < <(find "$DATA_CONFIG_DIR" -maxdepth 1 -type f -name '*.yaml' | sort)
mapfile -t MODEL_CONFIGS < <(find "$MODEL_CONFIG_DIR" -maxdepth 1 -type f -name '*.yaml' | sort)
mapfile -t LOSS_CONFIGS < <(find "$LOSS_CONFIG_DIR" -maxdepth 1 -type f -name '*.yaml' | sort)

if (( ${#DATA_CONFIGS[@]} == 0 || ${#MODEL_CONFIGS[@]} == 0 || ${#LOSS_CONFIGS[@]} == 0 )); then
  echo "No experiment configuration was found."
  exit 1
fi

declare -a GPU_PIDS
declare -a GPU_LOGS
FREE_GPU=-1

for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
  GPU_PIDS[$gpu]=""
  GPU_LOGS[$gpu]=""
done

cleanup() {
  for pid in "${GPU_PIDS[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT

release_finished_jobs() {
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    pid=${GPU_PIDS[$gpu]}
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      if ! wait "$pid"; then
        echo "Experiment failed: ${GPU_LOGS[$gpu]}"
        exit 1
      fi
      GPU_PIDS[$gpu]=""
      GPU_LOGS[$gpu]=""
    fi
  done
}

wait_for_gpu() {
  while true; do
    release_finished_jobs
    for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
      if [[ -z "${GPU_PIDS[$gpu]}" ]]; then
        FREE_GPU=$gpu
        return
      fi
    done
    sleep 1
  done
}

mkdir -p "$LOG_DIR"

for model_config in "${MODEL_CONFIGS[@]}"; do
  for loss_config in "${LOSS_CONFIGS[@]}"; do
    for data_config in "${DATA_CONFIGS[@]}"; do
      wait_for_gpu
      project=$(basename "$data_config" .yaml)
      model=$(basename "$model_config" .yaml)
      loss=$(basename "$loss_config" .yaml)
      log_path="$LOG_DIR/${project}_${model}_${loss}.log"
      echo "[RUN] GPU $FREE_GPU -> $log_path"
      CUDA_VISIBLE_DEVICES="$FREE_GPU" python -m scripts.train \
        --base_config "$BASE_CONFIG" \
        --data_config "$data_config" \
        --model_config "$model_config" \
        --loss_config "$loss_config" \
        >"$log_path" 2>&1 &
      GPU_PIDS[$FREE_GPU]=$!
      GPU_LOGS[$FREE_GPU]=$log_path
    done
  done
done

for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
  pid=${GPU_PIDS[$gpu]}
  if [[ -n "$pid" ]] && ! wait "$pid"; then
    echo "Experiment failed: ${GPU_LOGS[$gpu]}"
    exit 1
  fi
done

echo "All experiments finished."
