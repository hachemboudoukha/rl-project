import numpy as np

class EpsilonGreedyPolicy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, q_values, actions):
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)
        else:
            # Handle multiple actions with same max value
            max_q = np.max(q_values)
            best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
            return np.random.choice(best_actions)

    def update_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon
