import sys
import os

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from rl.environments.secret_env import SecretEnvWrapper
from rl.algorithms.temporal_difference.q_learning import QLearning
from rl.experiments.run_experiment import run_experiment

def test_secret_env(env_id):
    print(f"\n--- Testing Secret Environment {env_id} ---")
    try:
        env = SecretEnvWrapper(env_id=env_id)
        print(f"Number of states: {env.n_states}")
        print(f"Number of actions: {env.n_actions}")
        
        # 1. Random episode
        print("\nRunning a random episode:")
        state = env.reset()
        env.render()
        done = False
        steps = 0
        while not done and steps < 10:
            action = 0 # Simple action choice for demo
            state, reward, done, info = env.step(action)
            print(f"Step {steps+1}: Action {action} -> Reward {reward}, Done {done}")
            steps += 1
        
        # 2. Short Training (Smoke Test)
        print("\nRunning a short Q-Learning training (100 episodes):")
        agent = QLearning(env, alpha=0.1, gamma=0.9, epsilon=0.1)
        policy, Q = run_experiment(agent, episodes=100, name=f"secret_env_{env_id}_test")
        print(f"Training finished. Policy states: {len(policy)}")

    except Exception as e:
        print(f"Error testing Secret Env {env_id}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        env_ids = [int(sys.argv[1])]
    else:
        env_ids = [0, 1, 2, 3] # Test all by default
        
    for eid in env_ids:
        test_secret_env(eid)
