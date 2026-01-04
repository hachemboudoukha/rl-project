import sys
import os

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Environments
from rl.environments.line_world import LineWorld
from rl.environments.grid_world import GridWorld
from rl.environments.rock_paper_scissors import RockPaperScissors
from rl.environments.monty_hall_lvl1 import MontyHallLvl1
from rl.environments.monty_hall_lvl2 import MontyHallLvl2
from rl.environments.secret_env import SecretEnvWrapper

# Algorithms
from rl.algorithms.dynamic_programming.policy_iteration import PolicyIteration
from rl.algorithms.dynamic_programming.value_iteration import ValueIteration
from rl.algorithms.monte_carlo.mc_es import MonteCarloES
from rl.algorithms.monte_carlo.on_policy_mc import OnPolicyFirstVisitMC
from rl.algorithms.monte_carlo.off_policy_mc import OffPolicyMC
from rl.algorithms.temporal_difference.sarsa import SARSA
from rl.algorithms.temporal_difference.q_learning import QLearning
from rl.algorithms.temporal_difference.expected_sarsa import ExpectedSARSA
from rl.algorithms.planning.dyna_q import DynaQ
from rl.algorithms.planning.dyna_q_plus import DynaQPlus

from tqdm import tqdm
from rl.experiments.run_experiment import run_experiment

def train_all_on_env(env, env_name, episodes_td=5000, episodes_mc=10000):
    print(f"\n{'='*20} Training on {env_name} {'='*20}")
    
    # 1. Dynamic Programming (if applicable)
    if hasattr(env, 'get_states') and hasattr(env, 'state'):
        print("\n--- Running Dynamic Programming ---")
        pi = PolicyIteration(env)
        run_experiment(pi, episodes=200, name=f"pi_{env_name}", 
                       save_path=f"saved_models/policies/pi_{env_name}.pkl")
        
        vi = ValueIteration(env)
        run_experiment(vi, episodes=200, name=f"vi_{env_name}", 
                       save_path=f"saved_models/policies/vi_{env_name}.pkl")

    # 2. Monte Carlo
    #print("\n--- Running Monte Carlo ---")
    #mc_es = MonteCarloES(env)
    #run_experiment(mc_es, episodes=episodes_mc, name=f"mc_es_{env_name}", 
    #               save_path=f"saved_models/q_values/mc_es_{env_name}.pkl")
    
    mc_on = OnPolicyFirstVisitMC(env)
    run_experiment(mc_on, episodes=episodes_mc, name=f"mc_on_{env_name}", 
                   save_path=f"saved_models/q_values/mc_on_{env_name}.pkl")

    # 3. Temporal Difference
    print("\n--- Running Temporal Difference ---")
    sarsa = SARSA(env)
    run_experiment(sarsa, episodes=episodes_td, name=f"sarsa_{env_name}", 
                   save_path=f"saved_models/q_values/sarsa_{env_name}.pkl")
    
    ql = QLearning(env)
    run_experiment(ql, episodes=episodes_td, name=f"ql_{env_name}", 
                   save_path=f"saved_models/q_values/ql_{env_name}.pkl")

    # 4. Planning
    print("\n--- Running Planning ---")
    dyna = DynaQ(env)
    run_experiment(dyna, episodes=episodes_td, name=f"dyna_{env_name}", 
                   save_path=f"saved_models/q_values/dyna_{env_name}.pkl")

def main():
    print("🚀 Starting Global Training Experiment")
    
    # Environments to test
    envs = [
        (LineWorld(), "line_world"),
        (GridWorld(size=5), "grid_world"),
        (RockPaperScissors(), "rps"),
        (MontyHallLvl1(), "monty_hall_l1"),
        (MontyHallLvl2(), "monty_hall_l2")
    ]
    
    for env, name in tqdm(envs, desc="Global Progress (Envs)"):
        # We adjust episodes for speed in this demonstration
        # In a real experiment, you'd use more episodes
        train_all_on_env(env, name, episodes_td=1000, episodes_mc=2000)

    # Secret Environments (TD only for demo)
    print("\n🕵️ Training on Secret Environments")
    for i in tqdm(range(4), desc="Secret Envs"):
        try:
            env = SecretEnvWrapper(env_id=i)
            ql = QLearning(env)
            run_experiment(ql, episodes=1000, name=f"ql_secret_{i}", 
                           save_path=f"saved_models/q_values/ql_secret_{i}.pkl")
        except Exception as e:
            print(f"Error on Secret Env {i}: {e}")

    print("\n✅ All trainings completed!")

if __name__ == "__main__":
    main()
