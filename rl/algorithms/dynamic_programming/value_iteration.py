import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm
from ..base_agent import BaseAgent

class ValueIteration(BaseAgent):
    """
    Value Iteration Algorithm
    Requires a complete model of the environment.
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-6):
        super().__init__(env)
        self.gamma = gamma
        self.theta = theta
        self.V = {}
        self.policy = {}
    
    def train(self, max_iterations=1000) -> Tuple[Dict, Dict]:
        """Trains the agent using Value Iteration"""
        states = self.env.get_states()
        # Initialize V
        self.V = {s: 0.0 for s in states}
        
        pbar = tqdm(range(max_iterations), desc="Value Iteration")
        for iteration in pbar:
            delta = 0
            
            for state in states:
                if hasattr(self.env, 'is_terminal') and self.env.is_terminal(state):
                    continue
                
                v = self.V[state]
                
                # Calculate max_a Q(s,a)
                action_values = []
                for action in self.env.get_actions(state):
                    self.env.reset()
                    if hasattr(self.env, 'current_pos'):
                        self.env.current_pos = state
                    elif hasattr(self.env, 'state'):
                        self.env.state = state
                    
                    next_state, reward, done, _ = self.env.step(action)
                    action_value = reward + self.gamma * self.V[next_state]
                    action_values.append(action_value)
                
                if action_values:
                    self.V[state] = max(action_values)
                delta = max(delta, abs(v - self.V[state]))
            
            if delta < self.theta:
                pbar.set_postfix({"status": "converged"})
                break
        
        # Extract optimal policy
        
        # Extract optimal policy
        for state in states:
            if hasattr(self.env, 'is_terminal') and self.env.is_terminal(state):
                actions = self.env.get_actions(state)
                self.policy[state] = actions[0] if actions else 0
                continue
            
            action_values = []
            for action in self.env.get_actions(state):
                self.env.reset()
                if hasattr(self.env, 'current_pos'):
                    self.env.current_pos = state
                elif hasattr(self.env, 'state'):
                    self.env.state = state
                    
                next_state, reward, done, _ = self.env.step(action)
                action_value = reward + self.gamma * self.V[next_state]
                action_values.append((action, action_value))
            
            if action_values:
                self.policy[state] = max(action_values, key=lambda x: x[1])[0]
        
        return self.policy, self.V

    def act(self, state):
        return self.policy.get(state, np.random.choice(self.env.get_actions(state)))
