import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
from tqdm import tqdm
from ..base_agent import BaseAgent

class MonteCarloES(BaseAgent):
    """
    Monte Carlo Exploring Starts
    """
    
    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self.gamma = gamma
        self.Q = defaultdict(lambda: defaultdict(float))
        self.returns = defaultdict(lambda: defaultdict(list))
        self.policy = {}
    
    def generate_episode(self, start_state, start_action):
        """Generates an episode with exploring starts"""
        episode = []
        
        # Force the start
        self.env.reset()
        if hasattr(self.env, 'current_pos'):
            self.env.current_pos = start_state
        elif hasattr(self.env, 'state'):
            self.env.state = start_state
            
        state = start_state
        action = start_action
        
        # First step
        next_state, reward, done, _ = self.env.step(action)
        episode.append((state, action, reward))
        state = next_state
        
        # Follow the policy
        while not done:
            if state in self.policy:
                action = self.policy[state]
            else:
                action = np.random.choice(self.env.get_actions(state))
            
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        return episode
    
    def train(self, episodes=10000) -> Tuple[Dict, Dict]:
        """Trains the agent using MC ES"""
        states = self.env.get_states()
        # Initialize policy randomly
        for state in states:
            if hasattr(self.env, 'is_terminal') and self.env.is_terminal(state):
                continue
            self.policy[state] = np.random.choice(self.env.get_actions(state))
        
        for episode_num in tqdm(range(episodes), desc="MC ES"):
            # Exploring starts
            non_terminal_states = [s for s in states if not (hasattr(self.env, 'is_terminal') and self.env.is_terminal(s))]
            if not non_terminal_states:
                break
                
            start_state = np.random.choice(non_terminal_states)
            start_action = np.random.choice(self.env.get_actions(start_state))
            
            # Generate an episode
            episode = self.generate_episode(start_state, start_action)
            
            # Calculate returns
            G = 0
            visited = set()
            
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                # First-visit MC
                if (state, action) not in visited:
                    visited.add((state, action))
                    self.returns[state][action].append(G)
                    self.Q[state][action] = np.mean(self.returns[state][action])
                    
                    # Policy improvement
                    best_action = max(self.Q[state].items(), key=lambda x: x[1])[0]
                    self.policy[state] = best_action
        
        return self.policy, dict(self.Q)
        
        return self.policy, dict(self.Q)

    def act(self, state):
        return self.policy.get(state, np.random.choice(self.env.get_actions(state)))
