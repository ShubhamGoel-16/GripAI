#!/usr/bin/env bash
set -eu
set -o pipefail 2>/dev/null || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v apt-get >/dev/null 2>&1; then
  SUDO=""
  if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
      SUDO="sudo"
    else
      echo "sudo not found. Run as root or install sudo."
      exit 1
    fi
  fi
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -y
  $SUDO apt-get install -y python3 python3-venv python3-pip build-essential
fi

PYTHON_BIN=""
for candidate in python3.11 python3.10 python3.9 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" - <<'PY'
import sys
major, minor = sys.version_info[:2]
raise SystemExit(0 if (major, minor) >= (3, 9) and (major, minor) < (3, 12) else 1)
PY
    then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "No supported Python found. Install Python 3.9-3.11 and retry."
  exit 1
fi

if [ -d ".venv" ]; then
  if [ -x ".venv/bin/python" ]; then
    if ! ".venv/bin/python" - <<'PY'
import sys
major, minor = sys.version_info[:2]
raise SystemExit(0 if (major, minor) >= (3, 9) and (major, minor) < (3, 12) else 1)
PY
    then
      rm -rf .venv
    fi
  else
    rm -rf .venv
  fi
fi

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

python -m pip install --upgrade pip

# pybullet does not publish wheels for Python 3.12+ yet
python - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) >= (3, 12):
  print("ERROR: Python 3.12+ detected. Use Python 3.10 or 3.11 for pybullet wheels.")
  raise SystemExit(1)
PY

# Core dependencies for the pipeline
python -m pip install --only-binary=:all: pybullet
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install gymnasium panda-gym stable-baselines3 openai python-dotenv matplotlib tensorboard

export MPLBACKEND=Agg
export GRIPAI_HEADLESS=1
export GRIPAI_OFFLINE=1
export GRIPAI_BATCH=1

mkdir -p results run_logs

python llm_brain_with_training.py --batch --headless --offline --commands "reach the target;push the block;pick and place" | tee run_logs/llm_brain_with_training.log
python graphs.py | tee run_logs/graphs.log
