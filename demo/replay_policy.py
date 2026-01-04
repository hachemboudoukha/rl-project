import sys
import os
import time

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from rl.utils.serialization import load_object
from rl.environments.line_world import LineWorld
from rl.environments.grid_world import GridWorld
from rl.environments.rock_paper_scissors import RockPaperScissors
from rl.environments.monty_hall_lvl1 import MontyHallLvl1

def replay_policy(env, policy, delay=0.5):
    state = env.reset()
    env.render()
    done = False
    total_reward = 0
    
    while not done:
        time.sleep(delay)
        action = policy.get(state, 0) # Default to 0 if not in policy
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        print(f"Action: {action}, Reward: {reward}, Total Reward: {total_reward}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python replay_policy.py <env_name> <policy_path>")
        print("env_names: lineworld, gridworld, rps, monty1")
        sys.exit(1)
        
    env_name = sys.argv[1]
    policy_path = sys.argv[2]
    
    if env_name == "lineworld":
        env = LineWorld()
    elif env_name == "gridworld":
        env = GridWorld()
    elif env_name == "rps":
        env = RockPaperScissors()
    elif env_name == "monty1":
        env = MontyHallLvl1()
    else:
        print("Unknown environment.")
        sys.exit(1)
        
    try:
        data = load_object(policy_path)
        policy = data['policy']
        replay_policy(env, policy)
    except Exception as e:
        print(f"Error loading policy: {e}")
