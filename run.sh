#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3.9+ and retry."
  exit 1
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

python -m pip install --upgrade pip

# Core dependencies for the pipeline
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install gymnasium panda-gym stable-baselines3 openai python-dotenv matplotlib tensorboard

export MPLBACKEND=Agg
export GRIPAI_HEADLESS=1
export GRIPAI_OFFLINE=1
export GRIPAI_BATCH=1

mkdir -p results run_logs

python llm_brain_with_training.py --batch --headless --offline --commands "reach the target;push the block;pick and place" | tee run_logs/llm_brain_with_training.log
python graphs.py | tee run_logs/graphs.log
