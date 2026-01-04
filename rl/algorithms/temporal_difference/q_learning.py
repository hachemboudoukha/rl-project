import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
from tqdm import tqdm
from ..base_agent import BaseAgent
from ...policies.epsilon_greedy import EpsilonGreedyPolicy

class QLearning(BaseAgent):
    """
    Q-Learning
    Off-policy TD Control
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(env)
        self.alpha = alpha
        self.gamma = gamma
        self.policy_helper = EpsilonGreedyPolicy(epsilon)
        self.Q = defaultdict(lambda: defaultdict(float))
        self.policy = {}
    
    def train(self, episodes=5000) -> Tuple[Dict, Dict]:
        """Trains the agent using Q-Learning"""
        for episode_num in tqdm(range(episodes), desc="Q-Learning"):
            state = self.env.reset()
            done = False
            
            while not done:
                actions = self.env.get_actions(state)
                q_values = [self.Q[state][a] for a in actions]
                action = self.policy_helper.select_action(q_values, actions)
                
                next_state, reward, done, _ = self.env.step(action)
                
                # Q-Learning update (off-policy: use max)
                if done:
                    target = reward
                else:
                    max_q = max(self.Q[next_state].values()) if self.Q[next_state] else 0.0
                    target = reward + self.gamma * max_q
                
                self.Q[state][action] += self.alpha * (target - self.Q[state][action])
                
                state = next_state
        
        # Extract greedy policy
        
        # Extract greedy policy
        for state in self.Q:
            if self.Q[state]:
                self.policy[state] = max(self.Q[state].items(), key=lambda x: x[1])[0]
        
        return self.policy, dict(self.Q)

    def act(self, state):
        return self.policy.get(state, np.random.choice(self.env.get_actions(state)))
