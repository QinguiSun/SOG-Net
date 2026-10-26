#!/usr/bin/env bash
# Run an explicit index or single-device model stage in the existing mlip env.
set -euo pipefail

stage="${1:-}"
case "$stage" in
  environment|index|smoke|converge|train|evaluate) ;;
  *) echo "Usage: bash examples/elementa/run_single_gpu.sh {environment|index|smoke|converge|train|evaluate}" >&2; exit 2 ;;
esac

project_dir="${PROJECT_DIR:-/work/home/einlovestan/projects/SOG-Net/CACE-SOG}"
export_dir="${EXPORT_DIR:-/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/export}"
database="${DATABASE:-/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3}"
run_dir="${RUN_DIR:-runs/elementa-sog}"
device="${DEVICE:-cuda}"
batch_size="${BATCH_SIZE:-8}"
epochs="${EPOCHS:-50}"
n_dl="${N_DL:-1}"
python_bin="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$project_dir"

channels=(sog_candidates lattice_auxiliary pbe_lowdim_auxiliary)
if [[ "${WITH_SPIN:-0}" == "1" ]]; then
  channels+=(spin_review)
  database="${DATABASE:-/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa-with-spin.sqlite3}"
  run_dir="${RUN_DIR:-runs/elementa-sog-with-spin}"
  case "$stage" in
    smoke|converge|train)
      if [[ "${SPIN_POLICY:-}" != "consistent-state" || -z "${SPIN_POLICY_NOTE:-}" ]]; then
        echo "Spin model stages require SPIN_POLICY=consistent-state and SPIN_POLICY_NOTE describing the reviewed state policy." >&2
        exit 2
      fi
      ;;
  esac
fi

run_model_stage() {
  if [[ "${WITH_SPIN:-0}" == "1" ]]; then
    "$python_bin" -m cace_sog_training "$@" --spin-policy consistent-state --spin-policy-note "$SPIN_POLICY_NOTE"
  else
    "$python_bin" -m cace_sog_training "$@"
  fi
}

case "$stage" in
  environment)
    mkdir -p runs/provenance
    "$python_bin" -m cace_sog_training environment --output runs/provenance/environment.json
    ;;
  index)
    args=(index --export-dir "$export_dir" --database "$database" --channels "${channels[@]}")
    if [[ "${INDEX_RESUME:-0}" == "1" ]]; then args+=(--resume); fi
    "$python_bin" -m cace_sog_training "${args[@]}"
    ;;
  smoke)
    run_model_stage smoke --database "$database" --channels "${channels[@]}" --output "${SMOKE_DIR:-runs/elementa-smoke}" --device "$device" --batch-size "${SMOKE_BATCH_SIZE:-2}" --steps "${SMOKE_STEPS:-10}"
    ;;
  converge)
    args=(converge --database "$database" --channels "${channels[@]}" --output "${CONVERGENCE_DIR:-runs/elementa-convergence}" --device "$device" --n-dl 2 1 0.5 --samples "${CONVERGENCE_SAMPLES:-4}")
    if [[ -n "${CONVERGENCE_CHECKPOINT:-}" ]]; then args+=(--checkpoint "$CONVERGENCE_CHECKPOINT"); fi
    run_model_stage "${args[@]}"
    ;;
  train)
    args=(train --database "$database" --channels "${channels[@]}" --run-dir "$run_dir" --device "$device" --batch-size "$batch_size" --epochs "$epochs" --n-dl "$n_dl")
    if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
    run_model_stage "${args[@]}"
    ;;
  evaluate)
    "$python_bin" -m cace_sog_training evaluate --database "$database" --checkpoint "${CHECKPOINT:-$run_dir/best.pt}" --split test --output "${TEST_DIR:-$run_dir/test}" --device "$device"
    ;;
esac
