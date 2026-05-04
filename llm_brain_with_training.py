import argparse
import os
import json
import re
import time
import gymnasium as gym
import panda_gym
import numpy as np
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

DIRECTION_OFFSETS = {
    "left": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    "right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "front": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "back": np.array([0.0, -1.0, 0.0], dtype=np.float32),
}

DISTANCE_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"

ALLOWED_METRE_UNITS = {"m", "meter", "meters", "metre", "metres"}
ALLOWED_CM_UNITS = {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}
ENABLE_PROGRESS_BAR = os.getenv("GRIPAI_PROGRESS_BAR") == "1"

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

def _format_vector(values):
    return np.array2string(np.asarray(values, dtype=np.float32), precision=3, separator=", ")

def _normalize_distance_to_metres(value: float, unit: str):
    unit = (unit or "").strip().lower().rstrip(".")
    if unit in ALLOWED_CM_UNITS:
        return value / 100.0, None
    if unit in ALLOWED_METRE_UNITS:
        return value, None
    if not unit:
        return value, "No distance unit was provided. Supported units are centimetres (cm) and metres (m); interpreting the value as metres."
    return value, f"Unsupported distance unit '{unit}'. Supported units are centimetres (cm) and metres (m); interpreting the value as metres."

def _parse_absolute_target(user_command: str):
    coordinate_pattern = re.compile(
        rf"\(\s*({DISTANCE_RE})\s*,\s*({DISTANCE_RE})\s*,\s*({DISTANCE_RE})\s*\)"
    )
    match = coordinate_pattern.search(user_command)
    if not match:
        labelled_pattern = re.compile(
            rf"\bx\s*=\s*({DISTANCE_RE})\s*,?\s*y\s*=\s*({DISTANCE_RE})\s*,?\s*z\s*=\s*({DISTANCE_RE})\b",
            re.IGNORECASE,
        )
        match = labelled_pattern.search(user_command)
    if not match:
        return None

    return {
        "mode": "absolute",
        "target": [float(match.group(1)), float(match.group(2)), float(match.group(3))],
    }

def _parse_relative_target(user_command: str):
    directions = "|".join(DIRECTION_OFFSETS.keys())
    patterns = [
        re.compile(
            rf"\b(?P<distance>{DISTANCE_RE})\s*(?P<unit>[a-zA-Z]*)\s*(?:to\s+the\s+|to\s+|towards\s+the\s+|towards\s+)?(?P<direction>{directions})\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?P<direction>{directions})\b\s*(?:by|for)?\s*(?P<distance>{DISTANCE_RE})\s*(?P<unit>[a-zA-Z]*)\b",
            re.IGNORECASE,
        ),
    ]

    for pattern in patterns:
        match = pattern.search(user_command)
        if not match:
            continue
        distance_value = float(match.group("distance"))
        unit = match.group("unit")
        distance_m, warning = _normalize_distance_to_metres(distance_value, unit)
        return {
            "mode": "relative",
            "direction": match.group("direction").lower(),
            "distance_m": distance_m,
            "raw_distance": distance_value,
            "raw_unit": unit or "",
            "warning": warning,
        }
    return None

def _target_spec_from_tool_args(function_args: dict):
    target_position = function_args.get("target_position")
    if isinstance(target_position, list) and len(target_position) == 3:
        try:
            return {"mode": "absolute", "target": [float(v) for v in target_position]}
        except (TypeError, ValueError):
            return None

    relative_move = function_args.get("relative_move")
    if isinstance(relative_move, dict):
        direction = str(relative_move.get("direction", "")).lower()
        if direction not in DIRECTION_OFFSETS:
            return None
        try:
            distance_value = float(relative_move.get("distance"))
        except (TypeError, ValueError):
            return None
        distance_m, warning = _normalize_distance_to_metres(distance_value, str(relative_move.get("unit", "")))
        return {
            "mode": "relative",
            "direction": direction,
            "distance_m": distance_m,
            "raw_distance": distance_value,
            "raw_unit": str(relative_move.get("unit", "")),
            "warning": warning,
        }

    return None

def _extract_target_spec(user_command: str, function_args=None):
    # Deterministic parsing keeps coordinates and units consistent in offline mode.
    return (
        _parse_absolute_target(user_command)
        or _parse_relative_target(user_command)
        or _target_spec_from_tool_args(function_args or {})
    )

def _object_surface_goal_z(env):
    object_size = getattr(env.unwrapped.task, "object_size", 0.04)
    return float(object_size) / 2.0

def _apply_manipulation_target(env, obs, target_spec, skill_name: str):
    if not target_spec:
        print(f"🎯 BRAIN: No explicit target found for {skill_name}; using Panda Gym's sampled target.")
        return obs

    if target_spec["mode"] == "absolute":
        target_pos = np.array(target_spec["target"], dtype=np.float32)
        if skill_name == "push":
            surface_z = _object_surface_goal_z(env)
            if not np.isclose(target_pos[2], surface_z):
                print(
                    "🎯 BRAIN: Push is a surface-only skill; using the provided x/y "
                    f"coordinates and setting z to the object surface ({surface_z:.3f} m)."
                )
                target_pos[2] = surface_z
        print(f"🎯 BRAIN: Using explicit {skill_name} target coordinates in metres: {_format_vector(target_pos)}")
    elif target_spec["mode"] == "relative":
        if target_spec.get("warning"):
            print(f"⚠️  BRAIN: {target_spec['warning']}")
        direction = target_spec["direction"]
        gripper_pos = np.array(env.unwrapped.robot.get_ee_position(), dtype=np.float32)
        target_pos = gripper_pos + DIRECTION_OFFSETS[direction] * float(target_spec["distance_m"])
        target_pos[2] = _object_surface_goal_z(env)
        print(
            f"🎯 BRAIN: Using gripper-relative {skill_name} target "
            f"({target_spec['distance_m']:.4f} m {direction}): {_format_vector(target_pos)}"
        )
    else:
        print("🎯 BRAIN: Target instruction was not understood; using Panda Gym's sampled target.")
        return obs

    env.unwrapped.task.goal = target_pos
    env.unwrapped.task.sim.set_base_pose("target", target_pos, np.array([0.0, 0.0, 0.0, 1.0]))
    obs["desired_goal"] = target_pos
    return obs

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

def execute_rl_skill(skill_name: str, target_spec=None, render: bool = True, sleep_time: float = 0.04):
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
    if skill_name in {"push", "pickandplace"}:
        obs = _apply_manipulation_target(active_env, obs, target_spec, skill_name)
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
        model.learn(total_timesteps=2000, reset_num_timesteps=False, progress_bar=ENABLE_PROGRESS_BAR)
        
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
                    },
                    "target_position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Optional absolute push or pick-and-place target coordinates [x, y, z] in metres, when the user gives coordinates."
                    },
                    "relative_move": {
                        "type": "object",
                        "description": "Optional gripper-relative push or pick-and-place movement. Use only when the user gives a direction and distance.",
                        "properties": {
                            "direction": {
                                "type": "string",
                                "enum": ["left", "right", "front", "back"],
                                "description": "left=-x, right=+x, front=+y, back=-y."
                            },
                            "distance": {
                                "type": "number",
                                "description": "The numeric distance value from the user's command."
                            },
                            "unit": {
                                "type": "string",
                                "description": "The unit exactly as the user wrote it. Supported units are cm and m; unsupported units will be interpreted as metres."
                            }
                        },
                        "required": ["direction", "distance", "unit"]
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
            target_spec = _extract_target_spec(user_command)
            print(f"🧠 BRAIN: (offline) routing to '{chosen_skill}'.")
            result = execute_rl_skill(chosen_skill, target_spec=target_spec, render=render, sleep_time=sleep_time)
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
            tool_choice="auto",
            parallel_tool_calls=False
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
                target_spec = _extract_target_spec(user_command, function_args)
                print(f"🧠 BRAIN: I have decided to use the '{chosen_skill}' RL policy.")
                
                # TRIGGER THE RL MUSCLES!
                result = execute_rl_skill(chosen_skill, target_spec=target_spec, render=render, sleep_time=sleep_time)
                print(f"✅ OUTCOME: {result}")
                return
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