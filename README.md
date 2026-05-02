# GripAI (LLM-Routed RL Control for Panda)

This project runs a Franka Emika Panda simulator with pre-trained SAC policies. An OpenAI model routes natural-language commands to low-level skills: reach, push, and pick-and-place. The main entry point is `llm_brain_with_training.py`.

## Prerequisites
- Python 3.9+
- An OpenAI API key
- A machine that can open a Panda Gym simulation window

## Setup
1) Clone the repo and create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install -U pip
pip install gymnasium panda-gym stable-baselines3 openai python-dotenv
```

3) Add your API key in a `.env` file at the repo root:

```env
OPENAI_API_KEY=your_key_here
```

## Model files
The main script loads these files by default:
- `panda_expert_muscles.zip`
- `panda_pickandplace_expert.zip`
- `best_model/push/best_model.zip`

If your files live elsewhere or have different names, update the `MODELS` dict in `llm_brain_with_training.py`.

## Run
```bash
python llm_brain_with_training.py
```

Then type commands such as:
- "reach the target"
- "push the block"
- "pick and place"

The script opens a simulator window. If a skill fails, it performs a short online learning phase and saves a new `*_online_updated.zip` model next to the original.

## Troubleshooting
- If you see `ModuleNotFoundError: No module named 'panda_gym'`, reinstall with `pip install panda-gym`.
- If you get OpenAI auth errors, confirm `.env` exists and `OPENAI_API_KEY` is set.
