import numpy as np
import random
from typing import Tuple, List, Optional
from .base_env import BaseEnvironment

class MontyHallLvl1(BaseEnvironment):
    """
    Monty Hall Level 1
    3 doors, 2 decisions: Choose a door, then Stay or Switch.
    """
    def __init__(self):
        self.doors = [0, 1, 2]
        self.actions = {0: "wait/pick", 1: "keep", 2: "switch"}
        self.n_actions = 3
        self.n_states = 3  # 0: Init, 1: After host opens door, 2: Terminal
        self.reset()

    def reset(self) -> int:
        self.car = random.choice(self.doors)
        self.first_choice = random.choice(self.doors)
        self.opened = None
        self.done = False
        self.current_state = 0
        return self.current_state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        if self.done:
            return 2, 0.0, True, {}

        # State 0: Initial choice made (randomly in reset), host opens a door.
        if self.current_state == 0:
            possible_to_open = [d for d in self.doors if d != self.first_choice and d != self.car]
            self.opened = random.choice(possible_to_open)
            self.current_state = 1
            return self.current_state, 0.0, False, {"opened": self.opened}

        # State 1: Keep or Switch
        if self.current_state == 1:
            if action == 1:  # Keep
                final_choice = self.first_choice
            else:  # Switch
                final_choice = [d for d in self.doors if d != self.first_choice and d != self.opened][0]
            
            self.current_state = 2
            self.done = True
            reward = 1.0 if final_choice == self.car else 0.0
            return self.current_state, reward, True, {"final": final_choice, "car": self.car}

    def get_actions(self, state: Optional[int] = None) -> List[int]:
        if state == 0:
            return [0]  # Just Wait/Progress
        if state == 1:
            return [1, 2]  # Keep or Switch
        return [0]

    def get_states(self) -> List[int]:
        return [0, 1, 2]

    def is_terminal(self, state: int) -> bool:
        return state == 2

    def render(self):
        if self.current_state == 0:
            print(f"Phase 1: Agent initially chose door {self.first_choice}")
        elif self.current_state == 1:
            print(f"Phase 2: Host opened door {self.opened}. Choice: {self.first_choice}. KEEP(1) or SWITCH(2)?")
        elif self.current_state == 2:
            print(f"End: Car was behind {self.car}. Goal was reached: {self.done}")

    @property
    def state(self):
        return self.current_state

    @state.setter
    def state(self, value):
        self.current_state = value
        if value == 2:
            self.done = True
        else:
            self.done = False
