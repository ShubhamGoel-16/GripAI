import os
import json
import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from dotenv import load_dotenv
from openai import OpenAI

# Load the API key from the .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── 1. DEFINE YOUR RL MUSCLES ─────────────────────────────────────────────────
# Point these to the exact paths of your fully trained .zip models
MODELS = {
"reach": "./panda_expert_muscles.zip",               
    "push": "./best_model/push/best_model.zip",           
    "pickandplace": "./panda_pickandplace_expert.zip"}

def execute_rl_skill(skill_name: str, render: bool = True):
    """This function acts as the bridge between the LLM and your RL models."""
    print(f"\n⚙️  SYSTEM: Initializing RL Expert for '{skill_name}'...")
    
    # Map the requested skill to the correct environment
    if skill_name == "reach":
        env_id = "PandaReachDense-v3"
    elif skill_name == "push":
        env_id = "PandaPush-v3"
    elif skill_name == "pickandplace":
        env_id = "PandaPickAndPlace-v3"
    else:
        return f"Error: Skill '{skill_name}' not recognized."

    # Load the environment and the trained brain
    env = gym.make(env_id, render_mode="human" if render else None)
    model = SAC.load(MODELS[skill_name], env=env)
    
    obs, info = env.reset()
    done = False
    truncated = False
    
    print(f"🤖 ROBOT: Executing {skill_name} maneuver...")
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
    env.close()
    
    if info.get("is_success"):
        return f"Success: The {skill_name} task was completed perfectly."
    else:
        return f"Failed: The {skill_name} task was not completed."

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
def process_human_command(user_command: str):
    print(f"\n🗣️  HUMAN: {user_command}")
    print("🧠 BRAIN: Thinking...")

    response = client.chat.completions.create(
        model="gpt-4o-mini", # Fast, cheap, and smart enough for routing
        messages=[
            {"role": "system", "content": "You are GripAI, the high-level semantic planner for a Franka Emika Panda robotic arm. Your job is to translate natural language commands from humans into specific low-level robotic skills: 'reach', 'push', or 'pickandplace'. Call the tool to execute the action."},
            {"role": "user", "content": user_command}
        ],
        tools=tools,
        tool_choice="auto"
    )

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
                result = execute_rl_skill(chosen_skill)
                print(f"✅ OUTCOME: {result}")
    else:
        print(f"🧠 BRAIN: {response_message.content}")

# ── 4. TEST IT OUT ────────────────────────────────────────────────────────────
# ── 4. LIVE INTERACTIVE TERMINAL ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=========================================================")
    print("🤖 GripAI System Initialized.")
    print("Type your commands below. Type 'exit' or 'quit' to stop.")
    print("=========================================================")
    
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
            process_human_command(user_input)
        except Exception as e:
            print(f"❌ ERROR: Something went wrong communicating with the Brain or Muscles: {e}")