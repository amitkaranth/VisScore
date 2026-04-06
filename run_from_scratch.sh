#!/usr/bin/env bash
# VisScore: install deps, generate synthetic data, train CNN — from one script.
# Usage:
#   chmod +x run_from_scratch.sh
#   ./run_from_scratch.sh all|install|data|train|help
#
# Override defaults (examples):
#   DATA_DIR=./my_vis NUM_SAMPLES=200 EPOCHS=15 MODEL=efficientnet_b0 ./run_from_scratch.sh all

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- Configuration (override with env vars) ---
VENV_PATH="${VENV_PATH:-./.venv}"
PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-./vis_dataset}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
EPOCHS="${EPOCHS:-25}"
MODEL="${MODEL:-resnet50}"
AUGMENT_STRENGTH="${AUGMENT_STRENGTH:-medium}"

usage() {
  sed -n '1,12p' "$0" | tail -n +2
  echo "Commands:"
  echo "  all      — pip install, synthetic data, train (default)"
  echo "  install  — only: pip install -r requirements.txt"
  echo "  data     — only: synthetic_data_gen.py"
  echo "  train    — only: cnn_training.py (needs good/ and bad/ under DATA_DIR)"
  echo "  help     — this message"
  echo ""
  echo "See FROM_SCRATCH.md for data layout, VLMs, and Streamlit apps."
}

activate_venv() {
  if [[ -f "$VENV_PATH/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_PATH/bin/activate"
    echo "Using virtualenv: $VENV_PATH"
  else
    echo "Note: no venv at $VENV_PATH — using system $PYTHON"
  fi
}

cmd_install() {
  if [[ ! -f "requirements.txt" ]]; then
    echo "Error: requirements.txt not found in $ROOT"
    exit 1
  fi
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -r requirements.txt
  echo "Install done."
}

cmd_data() {
  "$PYTHON" synthetic_data_gen.py \
    --output_dir "$DATA_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --augment_strength "$AUGMENT_STRENGTH"
  echo "Synthetic data written under: $DATA_DIR"
}

cmd_train() {
  if [[ ! -d "$DATA_DIR/good" ]] || [[ ! -d "$DATA_DIR/bad" ]]; then
    echo "Error: expected $DATA_DIR/good and $DATA_DIR/bad (run 'data' first or set DATA_DIR)."
    exit 1
  fi
  "$PYTHON" cnn_training.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$RESULTS_DIR" \
    --model "$MODEL" \
    --epochs "$EPOCHS"
  echo "Training done. Weights: $RESULTS_DIR/best_model.pth"
}

cmd_all() {
  cmd_install
  cmd_data
  cmd_train
  echo ""
  echo "=== Next steps ==="
  echo "  streamlit run app.py                    # main UI (CNN + optional VLM)"
  echo "  streamlit run streamlit_csv_charts.py   # CSV → chart PNGs only"
  echo "  python inference.py --image <png> --model_path $RESULTS_DIR/best_model.pth --model_name $MODEL"
  echo "Full guide: $ROOT/FROM_SCRATCH.md"
}

main() {
  case "${1:-all}" in
    help|-h|--help) usage ;;
    install) activate_venv; cmd_install ;;
    data) activate_venv; cmd_data ;;
    train) activate_venv; cmd_train ;;
    all) activate_venv; cmd_all ;;
    *)
      echo "Unknown command: $1"
      usage
      exit 1
      ;;
  esac
}

main "$@"
