from abc import ABC, abstractmethod
import numpy as np

class BaseEnvironment(ABC):
    """
    Abstract base class for all environments.
    Ensures a consistent interface for RL algorithms.
    """
    
    @abstractmethod
    def reset(self):
        """
        Resets the environment to an initial state.
        Returns:
            initial_state: The state of the environment after reset.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Performs an action in the environment.
        Args:
            action: The action to perform.
        Returns:
            next_state: The state after the action.
            reward: The reward obtained.
            done: Boolean, whether the episode has ended.
            info: Dictionary with additional information.
        """
        pass

    @abstractmethod
    def get_actions(self, state=None):
        """
        Returns the list of possible actions.
        """
        pass

    @abstractmethod
    def get_states(self):
        """
        Returns the list of all possible states (if discrete).
        """
        pass

    @abstractmethod
    def render(self):
        """
        Visualizes the environment.
        """
        pass
