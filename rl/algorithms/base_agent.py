from abc import ABC, abstractmethod
import os
import pickle

class BaseAgent(ABC):
    """
    Abstract base class for all RL agents.
    """

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def train(self, episodes, **kwargs):
        """
        Trains the agent on the environment.
        """
        pass

    @abstractmethod
    def act(self, state):
        """
        Chooses an action based on the learned policy.
        """
        pass

    def save(self, path):
        """
        Saves the agent's state (policy, Q-values, etc.) to a file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path):
        """
        Loads the agent's state from a file.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.__dict__.update(state)
