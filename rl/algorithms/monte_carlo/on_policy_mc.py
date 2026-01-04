import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
from tqdm import tqdm
from ..base_agent import BaseAgent
from ...policies.epsilon_greedy import EpsilonGreedyPolicy

class OnPolicyFirstVisitMC(BaseAgent):
    """
    On-policy First-visit Monte Carlo Control (epsilon-greedy)
    """
    
    def __init__(self, env, gamma=0.99, epsilon=0.1):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon_greedy = EpsilonGreedyPolicy(epsilon)
        self.Q = defaultdict(lambda: defaultdict(float))
        self.returns = defaultdict(lambda: defaultdict(list))
        self.policy = {}
    
    def generate_episode(self):
        """Generates an episode following the epsilon-greedy policy"""
        episode = []
        state = self.env.reset()
        done = False
        
        while not done:
            actions = self.env.get_actions(state)
            q_values = [self.Q[state][a] for a in actions]
            action = self.epsilon_greedy.select_action(q_values, actions)
            
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        return episode
    
    def train(self, episodes=10000) -> Tuple[Dict, Dict]:
        """Trains the agent using On-policy First-visit MC"""
        for episode_num in tqdm(range(episodes), desc="On-policy MC"):
            episode = self.generate_episode()
            
            G = 0
            visited = set()
            
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                if (state, action) not in visited:
                    visited.add((state, action))
                    self.returns[state][action].append(G)
                    self.Q[state][action] = np.mean(self.returns[state][action])
            
            # Extract greedy policy
            for state in self.Q:
                if self.Q[state]:
                    self.policy[state] = max(self.Q[state].items(), key=lambda x: x[1])[0]
        
        return self.policy, dict(self.Q)
        
        return self.policy, dict(self.Q)

    def act(self, state):
        return self.policy.get(state, np.random.choice(self.env.get_actions(state)))
