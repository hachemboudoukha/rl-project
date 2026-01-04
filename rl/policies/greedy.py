import numpy as np

class GreedyPolicy:
    def select_action(self, q_values, actions):
        max_q = np.max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return np.random.choice(best_actions)
