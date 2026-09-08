#!/usr/bin/env bash
set -euo pipefail
cd /home/ywliang/ai4s/rec
method="$1"
gpu="$2"
code="$3"
case "$gpu" in
5) export CUDA_VISIBLE_DEVICES=GPU-302a7fe1-795f-752c-75b0-67be61c5e513 ;;
6) export CUDA_VISIBLE_DEVICES=GPU-7a4b9355-083c-c33a-a76b-454f0054cf64 ;;
*) exit 2 ;;
esac
root=runs/01_02/E006
exec 9>"runs/rec_atom_gpu$gpu.lock"
flock -n 9 || { echo 'GPU lock busy'; exit 3; }
active=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v uuid="$CUDA_VISIBLE_DEVICES" '$1 == uuid {print $2}')
if [ -n "$active" ]; then echo 'GPU has another process'; exit 3; fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/zhdd/home/ywliang/models/.hf_cache
export PYTHONPATH="/home/ywliang/ai4s/rec/$code/src"
printf '%s\n' "$BASHPID" > "$root/$method.pid"
if [ "$method" = features ]; then
  command=(/home/ywliang/miniconda3/envs/rec_atom/bin/python -u "$code/experiments/01_02/e006_groups.py" --config "$code/configs/01_02/e006_multidomain_methods.json" --phase features --gpu "$gpu")
else
  command=(/home/ywliang/miniconda3/envs/rec_atom/bin/python -u "$code/experiments/01_02/e006_train.py" --config "$code/configs/01_02/e006_multidomain_methods.json" --method "$method" --gpu "$gpu")
fi
if "${command[@]}"; then
  touch "$root/$method.finished"
else
  status=$?
  printf '%s\n' "$status" > "$root/$method.failed"
  exit "$status"
fi
