import numpy as np
from typing import Tuple, List, Optional
from .base_env import BaseEnvironment

class GridWorld(BaseEnvironment):
    """
    Grid World Environment
    The agent moves on a 2D grid.
    - Actions: 0 (up), 1 (right), 2 (down), 3 (left)
    - Reward: 1 at goal, 0 otherwise
    """
    
    def __init__(self, size: int = 5):
        self.size = size
        self.height = size
        self.width = size
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        
        self.reset()
        
        self.actions = {0: "up", 1: "right", 2: "down", 3: "left"}
        self.n_actions = len(self.actions)
        self.n_states = size * size
    
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.width + pos[1]
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        return (state // self.width, state % self.width)
    
    def reset(self) -> int:
        self.current_pos = self.start_pos
        self.done = False
        return self._pos_to_state(self.current_pos)
    
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        if self.done:
            return self._pos_to_state(self.current_pos), 0.0, True, {}
        
        row, col = self.current_pos
        if action == 0:  # Up
            new_pos = (max(0, row - 1), col)
        elif action == 1:  # Right
            new_pos = (row, min(self.width - 1, col + 1))
        elif action == 2:  # Down
            new_pos = (min(self.height - 1, row + 1), col)
        elif action == 3:  # Left
            new_pos = (row, max(0, col - 1))
        else:
            raise ValueError(f"Invalid action: {action}")
        
        self.current_pos = new_pos
        
        if self.current_pos == self.goal_pos:
            reward = 1.0
            self.done = True
        else:
            reward = 0.0
        
        info = {'position': self.current_pos}
        return self._pos_to_state(self.current_pos), reward, self.done, info
    
    def get_actions(self, state: Optional[int] = None) -> List[int]:
        return list(self.actions.keys())
    
    def get_states(self) -> List[int]:
        return list(range(self.n_states))
    
    def is_terminal(self, state: int) -> bool:
        return self._state_to_pos(state) == self.goal_pos
    
    def render(self):
        print("\n" + "=" * (self.width * 4 + 1))
        for row in range(self.height):
            line = "|"
            for col in range(self.width):
                pos = (row, col)
                if pos == self.current_pos and pos == self.goal_pos:
                    cell = " W "
                elif pos == self.current_pos:
                    cell = " A "
                elif pos == self.goal_pos:
                    cell = " G "
                else:
                    cell = " . "
                line += cell + "|"
            print(line)
            print("=" * (self.width * 4 + 1))
        print(f"Position: {self.current_pos} | Done: {self.done}\n")
