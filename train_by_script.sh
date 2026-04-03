#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SCRIPTS=(latin cyrillic arabic devanagari hangul zh ja)

SIZE="large"
DEVICE="cuda"
PROJECT="${WANDB_PROJECT:-cs489-canine}"
GROUP_PREFIX="${WANDB_GROUP_PREFIX:-by-script}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

usage() {
  cat <<'EOF'
Usage:
  ./train_by_script.sh [options] [script...]

Options:
  --size <name>       Dataset/config size to train. Default: large
  --device <name>     Training device override passed to train.py. Default: cuda
  --project <name>    W&B project name. Default: cs489-canine (or $WANDB_PROJECT)
  --group-prefix <s>  Prefix for W&B groups. Default: by-script
  --python <path>     Python executable to use. Default: .venv/bin/python
  -h, --help          Show this help

Examples:
  ./train_by_script.sh
  ./train_by_script.sh --size large --device cuda
  ./train_by_script.sh latin zh ja
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size)
      SIZE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --project)
      PROJECT="$2"
      shift 2
      ;;
    --group-prefix)
      GROUP_PREFIX="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  SCRIPTS=("$@")
else
  SCRIPTS=("${DEFAULT_SCRIPTS[@]}")
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No Python executable found. Set --python or PYTHON_BIN." >&2
    exit 1
  fi
fi

ORIGINAL_WANDB_TAGS="${WANDB_TAGS:-}"
WANDB_GROUP_VALUE="${WANDB_GROUP:-${GROUP_PREFIX}-${SIZE}}"

cd "$ROOT_DIR"

for script_name in "${SCRIPTS[@]}"; do
  config_path="configs/by_script/${script_name}/${SIZE}.json"
  if [[ ! -f "$config_path" ]]; then
    echo "Missing config: $config_path" >&2
    exit 1
  fi

  run_name="${script_name}-${SIZE}"
  run_tags="by-script,${SIZE},${script_name}"
  if [[ -n "$ORIGINAL_WANDB_TAGS" ]]; then
    run_tags="${ORIGINAL_WANDB_TAGS},${run_tags}"
  fi

  echo "==> Training ${script_name} with ${config_path}"
  echo "    W&B project=${PROJECT} group=${WANDB_GROUP_VALUE} name=${run_name}"

  WANDB_PROJECT="$PROJECT" \
  WANDB_GROUP="$WANDB_GROUP_VALUE" \
  WANDB_NAME="$run_name" \
  WANDB_TAGS="$run_tags" \
  PYTHONPATH=src \
  "$PYTHON_BIN" src/train.py --config "$config_path" --device "$DEVICE" --wandb
done
