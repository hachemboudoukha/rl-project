import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
from tqdm import tqdm
from ..base_agent import BaseAgent
from ...policies.epsilon_greedy import EpsilonGreedyPolicy

class SARSA(BaseAgent):
    """
    SARSA (State-Action-Reward-State-Action)
    On-policy TD Control
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(env)
        self.alpha = alpha
        self.gamma = gamma
        self.policy_helper = EpsilonGreedyPolicy(epsilon)
        self.Q = defaultdict(lambda: defaultdict(float))
        self.policy = {}
    
    def train(self, episodes=5000) -> Tuple[Dict, Dict]:
        """Trains the agent using SARSA"""
        for episode_num in tqdm(range(episodes), desc="SARSA"):
            state = self.env.reset()
            
            actions = self.env.get_actions(state)
            q_values = [self.Q[state][a] for a in actions]
            action = self.policy_helper.select_action(q_values, actions)
            
            done = False
            
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                
                next_actions = self.env.get_actions(next_state)
                next_q_values = [self.Q[next_state][a] for a in next_actions]
                next_action = self.policy_helper.select_action(next_q_values, next_actions)
                
                # SARSA update
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * self.Q[next_state][next_action]
                
                self.Q[state][action] += self.alpha * (target - self.Q[state][action])
                
                state = next_state
                action = next_action
        
        # Extract greedy policy
        
        # Extract greedy policy
        for state in self.Q:
            if self.Q[state]:
                self.policy[state] = max(self.Q[state].items(), key=lambda x: x[1])[0]
        
        return self.policy, dict(self.Q)

    def act(self, state):
        return self.policy.get(state, np.random.choice(self.env.get_actions(state)))
