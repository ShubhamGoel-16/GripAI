#!/usr/bin/env bash
set -eu
set -o pipefail 2>/dev/null || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_CMD=()

_set_python_cmd() {
  PYTHON_CMD=("$@")
}

_is_supported_python() {
  "${PYTHON_CMD[@]}" - <<'PY'
import sys
major, minor = sys.version_info[:2]
raise SystemExit(0 if (major, minor) >= (3, 9) and (major, minor) < (3, 12) else 1)
PY
}

# Prefer the default python if it is supported; otherwise look for 3.11/3.10/3.9.
if command -v python3 >/dev/null 2>&1; then
  _set_python_cmd python3
elif command -v python >/dev/null 2>&1; then
  _set_python_cmd python
fi

if [ ${#PYTHON_CMD[@]} -eq 0 ] || ! _is_supported_python; then
  PYTHON_CMD=()
  for candidate in python3.11 python3.10 python3.9; do
    if command -v "$candidate" >/dev/null 2>&1; then
      _set_python_cmd "$candidate"
      if _is_supported_python; then
        break
      fi
      PYTHON_CMD=()
    fi
  done
fi

if [ ${#PYTHON_CMD[@]} -eq 0 ] && command -v py >/dev/null 2>&1; then
  for candidate in 3.11 3.10 3.9; do
    _set_python_cmd py "-$candidate"
    if _is_supported_python; then
      break
    fi
    PYTHON_CMD=()
  done
fi

if [ ${#PYTHON_CMD[@]} -eq 0 ]; then
  echo "No supported Python 3.9-3.11 interpreter found. Install Python 3.11 or 3.10 and retry."
  exit 1
fi

if [ ! -d ".venv" ]; then
  "${PYTHON_CMD[@]}" -m venv .venv
fi

# shellcheck disable=SC1091
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
elif [ -f ".venv/Scripts/activate" ]; then
  source ".venv/Scripts/activate"
else
  echo "Virtual environment activation script not found."
  exit 1
fi

python -m pip install --upgrade pip

# pybullet does not publish wheels for Python 3.12+ yet
python - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 9) or (major, minor) >= (3, 12):
  print(f"ERROR: Python {major}.{minor} detected in .venv. This project needs Python 3.9-3.11 for pybullet wheels.")
  print("Delete .venv and re-run, or install Python 3.11 and re-run.")
  raise SystemExit(1)
PY

# Core dependencies for the pipeline
python -m pip install --only-binary=:all: pybullet
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install gymnasium panda-gym stable-baselines3 openai python-dotenv matplotlib tensorboard

export GRIPAI_OFFLINE=1

mkdir -p results run_logs

python llm_brain_with_training.py --offline | tee run_logs/llm_brain_with_training.log
python graphs.py | tee run_logs/graphs.log
