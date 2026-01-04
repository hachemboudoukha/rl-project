import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
from tqdm import tqdm
from ..base_agent import BaseAgent
from ...policies.epsilon_greedy import EpsilonGreedyPolicy

class OffPolicyMC(BaseAgent):
    """
    Off-policy Monte Carlo Control with Importance Sampling
    """
    
    def __init__(self, env, gamma=0.99, epsilon=0.1):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.behavior_policy_helper = EpsilonGreedyPolicy(epsilon)
        self.Q = defaultdict(lambda: defaultdict(float))
        self.C = defaultdict(lambda: defaultdict(float))  # Cumulative weights
        self.target_policy = {}  # Greedy policy
    
    def generate_episode(self):
        """Generates an episode with the behavior policy"""
        episode = []
        state = self.env.reset()
        done = False
        
        while not done:
            actions = self.env.get_actions(state)
            q_values = [self.Q[state][a] for a in actions]
            action = self.behavior_policy_helper.select_action(q_values, actions)
            
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        return episode
    
    def train(self, episodes=10000) -> Tuple[Dict, Dict]:
        """Trains the agent using Off-policy MC"""
        for episode_num in tqdm(range(episodes), desc="Off-policy MC"):
            episode = self.generate_episode()
            
            G = 0
            W = 1.0
            
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                self.C[state][action] += W
                self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])
                
                # Update target policy (greedy)
                if self.Q[state]:
                    best_action = max(self.Q[state].items(), key=lambda x: x[1])[0]
                    self.target_policy[state] = best_action
                    
                    # If action != best_action, stop
                    if action != best_action:
                        break
                
                # Importance sampling ratio
                # behavior: epsilon-greedy, target: greedy
                actions = self.env.get_actions(state)
                num_actions = len(actions)
                if action == self.target_policy.get(state, action):
                    W *= 1.0 / ((1 - self.epsilon) + self.epsilon / num_actions)
                else:
                    W *= 1.0 / (self.epsilon / num_actions)
        
        return self.target_policy, dict(self.Q)
        
        return self.target_policy, dict(self.Q)

    def act(self, state):
        return self.target_policy.get(state, np.random.choice(self.env.get_actions(state)))
