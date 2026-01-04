import sys
import os

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from rl.environments.line_world import LineWorld
from rl.environments.grid_world import GridWorld
from rl.environments.rock_paper_scissors import RockPaperScissors
from rl.environments.monty_hall_lvl1 import MontyHallLvl1

def play_manual(env):
    state = env.reset()
    env.render()
    done = False
    total_reward = 0
    
    while not done:
        try:
            action = int(input(f"Choose action {env.get_actions(state)}: "))
            state, reward, done, info = env.step(action)
            total_reward += reward
            env.render()
            print(f"Reward: {reward}, Total Reward: {total_reward}")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    print("Select environment:")
    print("1: Line World")
    print("2: Grid World")
    print("3: Rock Paper Scissors")
    print("4: Monty Hall Lvl 1")
    
    choice = input("Choice: ")
    if choice == "1":
        env = LineWorld()
    elif choice == "2":
        env = GridWorld()
    elif choice == "3":
        env = RockPaperScissors()
    elif choice == "4":
        env = MontyHallLvl1()
    else:
        print("Invalid choice.")
        sys.exit(1)
        
    play_manual(env)
