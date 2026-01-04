import numpy as np
import random
from typing import Tuple, List, Optional
from .base_env import BaseEnvironment

class MontyHallLvl2(BaseEnvironment):
    """
    Monty Hall Level 2
    5 doors, 4 decisions before final choice.
    This is a scalability test.
    """
    def __init__(self):
        self.doors = list(range(5))
        self.n_actions = 2  # 0: Keep, 1: Switch (in each step)
        # We model this as a sequence of steps where doors are opened one by one.
        # State can be: (current_choice, opened_doors_mask)
        # For simplicity in discrete RL, let's use a simpler state representation.
        self.reset()

    def reset(self) -> int:
        self.car = random.choice(self.doors)
        self.current_choice = random.choice(self.doors)
        self.opened_doors = set()
        self.done = False
        self.step_count = 0
        return self._get_state()

    def _get_state(self) -> int:
        # State consists of (step_count, choice_index)
        # Choice index is 0 to 4. Step count is 0 to 3.
        return self.step_count * 5 + self.current_choice

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        if self.done:
            return self._get_state(), 0.0, True, {}

        # 1. Host opens a door that is not the current choice and not the car
        possible_to_open = [d for d in self.doors if d != self.current_choice and d != self.car and d not in self.opened_doors]
        if possible_to_open:
            door_to_open = random.choice(possible_to_open)
            self.opened_doors.add(door_to_open)
        
        # 2. Agent chooses to Keep or Switch
        if action == 1: # Switch
            remaining = [d for d in self.doors if d != self.current_choice and d not in self.opened_doors]
            self.current_choice = random.choice(remaining)
        
        self.step_count += 1
        
        if self.step_count >= 4: # 4 decisions made
            self.done = True
            reward = 1.0 if self.current_choice == self.car else 0.0
            return self._get_state(), reward, True, {"car": self.car}
        
        return self._get_state(), 0.0, False, {}

    def get_actions(self, state: Optional[int] = None) -> List[int]:
        return [0, 1]

    def get_states(self) -> List[int]:
        # 4 steps * 5 doors
        return list(range(25))

    def is_terminal(self, state: int) -> bool:
        return state >= 20

    def render(self):
        print(f"Step {self.step_count}: doors opened={self.opened_doors}, current choice={self.current_choice}")
        if self.done:
            print(f"Result: Car was behind {self.car}. {'WIN' if self.current_choice == self.car else 'LOSE'}")
