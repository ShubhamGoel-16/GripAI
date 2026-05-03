import argparse
import os
import json
import time
import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from dotenv import load_dotenv
from openai import OpenAI

# Load the API key from the .env file
load_dotenv()
_openai_client = None

# ── 1. DEFINE YOUR RL MUSCLES ─────────────────────────────────────────────────
# Point these to the exact paths of your fully trained .zip models
MODELS = {
    "reach": "./panda_expert_muscles.zip",               
    "push": "./best_model/push/best_model.zip",           
    "pickandplace": "./panda_pickandplace_expert.zip"}

# Add these two exact lines right here!
active_env = None
active_env_key = None

def _get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    _openai_client = OpenAI(api_key=api_key)
    return _openai_client

def _route_command_offline(user_command: str):
    text = user_command.lower()
    if "pick" in text or "place" in text:
        return "pickandplace"
    if "push" in text:
        return "push"
    if "reach" in text or "move" in text or "target" in text:
        return "reach"
    return None

def _parse_command_list(raw: str):
    if not raw:
        return []
    for sep in [";", ","]:
        raw = raw.replace(sep, "\n")
    return [line.strip() for line in raw.splitlines() if line.strip()]

def _load_batch_commands(arg_value: str):
    env_value = os.getenv("GRIPAI_COMMANDS", "")
    source = arg_value or env_value
    if source and os.path.exists(source):
        with open(source, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    commands = _parse_command_list(source)
    if commands:
        return commands
    return ["reach the target", "push the block", "pick and place"]

def _close_active_env():
    global active_env, active_env_key
    if active_env is not None:
        active_env.close()
    active_env = None
    active_env_key = None

def execute_rl_skill(skill_name: str, render: bool = True, sleep_time: float = 0.04):
    global active_env, active_env_key
    
    print(f"\n⚙️  SYSTEM: Initializing RL Expert for '{skill_name}'...")
    
    if skill_name == "reach":
        env_id = "PandaReachDense-v3"
    elif skill_name == "push":
        env_id = "PandaPush-v3"
    elif skill_name == "pickandplace":
        env_id = "PandaPickAndPlace-v3"
    else:
        return f"Error: Skill '{skill_name}' not recognized."

    render_mode = "human" if render else "rgb_array"
    env_key = (env_id, render_mode)

    if active_env_key != env_key:
        if active_env is not None:
            active_env.close()
        active_env = gym.make(env_id, render_mode=render_mode)
        active_env_key = env_key

    # Load the specific expert brain
    model = SAC.load(MODELS[skill_name], env=active_env)
    
    obs, info = active_env.reset()
    done = False
    truncated = False
    
    print(f"🤖 ROBOT: Executing {skill_name} maneuver...")
    
    # --- PHASE 1: INFERENCE (The Attempt) ---
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = active_env.step(action)
        if sleep_time > 0:
            time.sleep(sleep_time)
        
    # --- PHASE 2: ONLINE LEARNING (The Self-Correction) ---
    if info.get("is_success"):
        return f"Success: The {skill_name} task was completed perfectly."
    else:
        print(f"❌ ROBOT: I failed the {skill_name} task. Initiating Online Learning Protocol...")
        
        # Turn off the sleep timer so it can train at maximum speed!
        # active_env.render_mode = None 
        model.learning_starts = model.num_timesteps + 100        # Let the robot practice and explore for 2,000 steps using its Replay Buffer
        model.learn(total_timesteps=2000, reset_num_timesteps=False, progress_bar=True)
        
        # Overwrite the old brain with the newly updated weights!
        # SAFE SAVE: Create a new file instead of destroying the original!
        safe_save_path = MODELS[skill_name].replace(".zip", "_online_updated.zip")
        model.save(safe_save_path)
        print(f"💾 SYSTEM: Neural weights safely saved to a new file: {safe_save_path}")
        print("💾 SYSTEM: Neural weights updated and saved. I will be smarter next time.")
        
        return f"Failed: The {skill_name} task was missed, but the agent performed 2000 steps of continuous online learning to update its weights."

# ── 2. DEFINE THE LLM TOOLS ───────────────────────────────────────────────────
# This tells the LLM exactly what its "Muscles" can do.
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_rl_skill",
            "description": "Executes a low-level robotic skill using a trained Reinforcement Learning policy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "enum": ["reach", "push", "pickandplace"],
                        "description": "The specific robotic skill to execute based on the user's command."
                    }
                },
                "required": ["skill_name"]
            }
        }
    }
]

# ── 3. THE LLM DECISION LOOP ──────────────────────────────────────────────────
def process_human_command(user_command: str, prefer_llm: bool = True, render: bool = True, sleep_time: float = 0.04):
    print(f"\n🗣️  HUMAN: {user_command}")
    print("🧠 BRAIN: Thinking...")

    def _run_offline():
        chosen_skill = _route_command_offline(user_command)
        if chosen_skill:
            print(f"🧠 BRAIN: (offline) routing to '{chosen_skill}'.")
            result = execute_rl_skill(chosen_skill, render=render, sleep_time=sleep_time)
            print(f"✅ OUTCOME: {result}")
        else:
            print("🧠 BRAIN: (offline) could not map the command to a skill.")
        return

    if not prefer_llm:
        return _run_offline()

    client = _get_openai_client()
    if client is None:
        print("🧠 BRAIN: OpenAI key missing. Falling back to offline routing.")
        return _run_offline()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Fast, cheap, and smart enough for routing
            messages=[
                {"role": "system", "content": "You are GripAI, the high-level semantic planner for a Franka Emika Panda robotic arm. Your job is to translate natural language commands from humans into specific low-level robotic skills: 'reach', 'push', or 'pickandplace'. Call the tool to execute the action."},
                {"role": "user", "content": user_command}
            ],
            tools=tools,
            tool_choice="auto"
        )
    except Exception as e:
        print(f"🧠 BRAIN: LLM call failed ({e}). Falling back to offline routing.")
        return _run_offline()

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "execute_rl_skill":
                chosen_skill = function_args.get("skill_name")
                print(f"🧠 BRAIN: I have decided to use the '{chosen_skill}' RL policy.")
                
                # TRIGGER THE RL MUSCLES!
                result = execute_rl_skill(chosen_skill, render=render, sleep_time=sleep_time)
                print(f"✅ OUTCOME: {result}")
    else:
        print(f"🧠 BRAIN: {response_message.content}")

# ── 4. TEST IT OUT ────────────────────────────────────────────────────────────
# ── 4. LIVE INTERACTIVE TERMINAL ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GripAI LLM-routed RL control.")
    parser.add_argument("--batch", action="store_true", help="Run a non-interactive batch and exit.")
    parser.add_argument("--commands", type=str, default="", help="Command list or path to a command file.")
    parser.add_argument("--headless", action="store_true", help="Disable GUI rendering.")
    parser.add_argument("--offline", action="store_true", help="Disable OpenAI routing.")
    args = parser.parse_args()

    headless = args.headless or os.getenv("GRIPAI_HEADLESS") == "1"
    offline = args.offline or os.getenv("GRIPAI_OFFLINE") == "1"
    if not os.getenv("OPENAI_API_KEY"):
        offline = True
    sleep_time = 0.0 if headless else 0.04
    run_batch = args.batch or os.getenv("GRIPAI_BATCH") == "1"

    print("=========================================================")
    print("🤖 GripAI System Initialized.")
    print("Type your commands below. Type 'exit' or 'quit' to stop.")
    print("=========================================================")

    if run_batch:
        commands = _load_batch_commands(args.commands)
        print(f"Running batch mode with {len(commands)} commands...")
        for command in commands:
            try:
                process_human_command(command, prefer_llm=not offline, render=not headless, sleep_time=sleep_time)
            except Exception as e:
                print(f"❌ ERROR: Batch command failed: {e}")
        _close_active_env()
        raise SystemExit(0)
    
    while True:
        # 1. Wait for the user to type something and press Enter
        user_input = input("\n🗣️  YOU: ")
        
        # 2. Check if the user wants to quit
        if user_input.lower() in ['exit', 'quit']:
            print("🛑 Shutting down GripAI. Goodbye!")
            break
            
        # 3. Skip if the user just pressed Enter without typing anything
        if not user_input.strip():
            continue
            
        # 4. Send the typed text to the LLM Brain!
        try:
            process_human_command(user_input, prefer_llm=not offline, render=not headless, sleep_time=sleep_time)
        except Exception as e:
            print(f"❌ ERROR: Something went wrong communicating with the Brain or Muscles: {e}")
