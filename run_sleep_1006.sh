#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RAW_ROOT=/home/ubuntu/deform-colon/data
PROC_ROOT="$REPO_ROOT/data/c3vd-v2"
RESULT_ROOT_BASE="$REPO_ROOT/OUTPUT"
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
GPU_INDEX="${GPU_ID%%,*}"

# SEQUENCES=(v4 v3)
SEQUENCES=(v4)
declare -A SCALED_DEPTHS=()

EXPERIMENTS=(
  # "bootstapir bootstapir bootstapir 16"
  "cotracker cotracker cotracker 32"
)

log_msg() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

wait_for_gpu() {
  local gpu_id="$1"
  local poll_seconds="${2:-300}"

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log_msg "nvidia-smi not found; proceeding without GPU availability check."
    return
  fi

  while true; do
    if ! mapfile -t active_pids < <(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d'); then
      log_msg "Unable to query GPU ${gpu_id}; proceeding without further checks."
      break
    fi
    if [[ ${#active_pids[@]} -eq 0 ]]; then
      log_msg "GPU ${gpu_id} is free."
      break
    fi
    log_msg "GPU ${gpu_id} busy (PIDs: ${active_pids[*]}). Sleeping ${poll_seconds}s..."
    sleep "$poll_seconds"
  done
}

run_cmd() {
  local label="$1"
  shift
  wait_for_gpu "$GPU_INDEX"
  log_msg "Starting: ${label}"
  ("$@")
  local status=$?
  if [[ $status -ne 0 ]]; then
    log_msg "WARNING: ${label} failed with exit code ${status}"
  else
    log_msg "Completed: ${label}"
  fi
}


for exp in "${EXPERIMENTS[@]}"; do
  read -r EXP_NAME TRACK_LABEL TRACK_MODEL GRID_SIZE <<<"$exp"
  RESULT_ROOT="${RESULT_ROOT_BASE}_${EXP_NAME}"
  log_msg "======= Experiment: ${EXP_NAME} ======="

  for seq in "${SEQUENCES[@]}"; do
    RAW_DIR="${RAW_ROOT}/${seq}"
    OUT_DIR="${PROC_ROOT}/${seq}"
    TRACK_DIR="${OUT_DIR}/${TRACK_LABEL}"
    DEPTH_DIR="${OUT_DIR}/depths"
    WORK_DIR="${RESULT_ROOT}/${seq}/${EXP_NAME}"

    mkdir -p "$OUT_DIR" "$WORK_DIR"

    custom_grid="$GRID_SIZE"
    if [[ "$seq" == "v3" ]]; then
      if [[ "$TRACK_MODEL" == "bootstapir" ]]; then
        custom_grid=32
      elif [[ "$TRACK_MODEL" == "cotracker" ]]; then
        custom_grid=16
      fi
    fi

    run_cmd "process_c3vd (${seq}, ${EXP_NAME})" \
      env CUDA_VISIBLE_DEVICES="$GPU_ID" python "$REPO_ROOT/preproc/process_c3vd.py" \
        --input_dir "$RAW_DIR" \
        --output_dir "$OUT_DIR" \
        --image_subdir rgb \
        --depth_subdir depth \
        --track_model "$TRACK_MODEL" \
        --track_out_name "$TRACK_LABEL" \
        --grid_size "$custom_grid" \
        --gpu "$GPU_ID"

    run_cmd "run_training (${seq}, ${EXP_NAME})" \
      env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/run_training.py" \
        --work-dir "$WORK_DIR" \
        --port 6010 \
        data:c3vd \
        --data.data-dir "$OUT_DIR" \
        --data.track_2d_type "$TRACK_LABEL"

    # run_cmd "evaluate_custom (${seq}, ${EXP_NAME})" \
    #   env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/scripts/evaluate_custom.py" \
    #     --data_dir "$PROC_ROOT" \
    #     --result_dir "$WORK_DIR" \
    #     --seq_names "$seq" \
    #     --image_subdir images \
    #     --depth_subdir depths \
    #     --tracks_subdir "$TRACK_LABEL" \
    #     --camera_file pose.txt \
    #     --intrinsics_file intrinsics.json
  done
done

log_msg "All experiments scheduled."


    # PYTHONPATH=/home/ubuntu/deform-colon/shape-of-motion python /home/ubuntu/deform-colon/shape-of-motion/run_training.py \
    #     --work-dir /home/ubuntu/deform-colon/shape-of-motion/OUTPUT_bootstapir/v4 \
    #     --port 6010 \
    #     data:c3vd \
    #     --data.data-dir /home/ubuntu/deform-colon/shape-of-motion/data/c3vd-v2/v4 \
    #     --data.track_2d_type bootstapir
