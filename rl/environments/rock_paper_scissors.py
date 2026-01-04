import numpy as np
from typing import Tuple, List, Optional
from .base_env import BaseEnvironment

class RockPaperScissors(BaseEnvironment):
    """
    Two Round Rock Paper Scissors Environment
    Opponent plays randomly in round 1, then repeats agent's round 1 choice in round 2.
    """
    
    def __init__(self):
        self.actions = {0: "Rock", 1: "Paper", 2: "Scissors"}
        self.n_actions = 3
        self.n_states = 11  # 0 (start), 1-9 (after round 1), 10 (terminal)
        self.reset()

    def reset(self) -> int:
        self.current_round = 0
        self.total_reward = 0
        self.done = False
        self.agent_r1 = None
        self.opp_r1 = None
        return 0

    def _encode_state(self) -> int:
        if self.done:
            return 10
        if self.current_round == 0:
            return 0
        # state = 1 + agent_r1 * 3 + opp_r1
        return 1 + self.agent_r1 * 3 + self.opp_r1

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        if self.done:
            return 10, 0.0, True, {}
        
        # Round 1: Opponent is random. Round 2: Opponent repeats Agent's Round 1.
        opp_action = np.random.choice(3) if self.current_round == 0 else self.agent_r1
        
        # Reward: +1 Win, -1 Loss, 0 Tie
        reward = float([0, 1, -1][(action - opp_action) % 3])
        self.total_reward += reward
        
        if self.current_round == 0:
            self.agent_r1, self.opp_r1 = action, opp_action
            self.current_round = 1
        else:
            self.done = True
            
        info = {
            "round": self.current_round,
            "agent_choice": self.actions[action],
            "opponent_choice": self.actions[opp_action],
            "reward": reward
        }
        
        return self._encode_state(), reward, self.done, info

    def get_actions(self, state: Optional[int] = None) -> List[int]:
        return [0, 1, 2]

    def get_states(self) -> List[int]:
        return list(range(11))

    def is_terminal(self, state: int) -> bool:
        return state == 10

    def render(self):
        if self.current_round == 0:
            print("\n--- Round 1: Choose 0:Rock, 1:Paper, 2:Scissors ---")
        elif self.done:
            print(f"--- Game Over! Score: {self.total_reward} ---")
        else:
            print(f"--- Round 2: Opponent's previous choice was {self.actions[self.opp_r1]} ---")
            print(f"--- Round 2: Opponent will play {self.actions[self.agent_r1]} (your R1 choice) ---")

    # Helper for DP algorithms (to teleport to a state)
    @property
    def state(self):
        return self._encode_state()

    @state.setter
    def state(self, value):
        if value == 0:
            self.reset()
        elif value == 10:
            self.done = True
        else:
            val = value - 1
            self.agent_r1 = val // 3
            self.opp_r1 = val % 3
            self.current_round = 1
            self.done = False
