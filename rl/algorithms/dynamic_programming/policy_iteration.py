import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm
from ..base_agent import BaseAgent

class PolicyIteration(BaseAgent):
    """
    Policy Iteration Algorithm
    Requires a complete model of the environment (P and R) or the ability to simulate any state-action.
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-6):
        super().__init__(env)
        self.gamma = gamma
        self.theta = theta
        self.V = {}  # Value function
        self.policy = {}  # Policy (deterministic)
        
    def policy_evaluation(self, policy: Dict) -> Dict:
        """Evaluates a given policy"""
        states = self.env.get_states()
        V = {s: 0.0 for s in states}
        
        while True:
            delta = 0
            for state in states:
                # We assume the env has a way to check if state is terminal
                # If not explicitly in BaseEnv, we might need to add it or handle it
                if hasattr(self.env, 'is_terminal') and self.env.is_terminal(state):
                    continue
                    
                v = V[state]
                action = policy[state]
                
                # Simulate the action to get next_state and reward
                # This approach assumes we can teleport the env to any state
                self.env.reset()
                if hasattr(self.env, 'current_pos'):
                    self.env.current_pos = state
                elif hasattr(self.env, 'state'):
                    self.env.state = state
                
                next_state, reward, done, _ = self.env.step(action)
                
                V[state] = reward + self.gamma * V[next_state]
                delta = max(delta, abs(v - V[state]))
            
            if delta < self.theta:
                break
        
        return V
    
    def policy_improvement(self, V: Dict) -> Tuple[Dict, bool]:
        """Improves the policy based on the value function"""
        policy_stable = True
        new_policy = {}
        states = self.env.get_states()
        
        for state in states:
            if hasattr(self.env, 'is_terminal') and self.env.is_terminal(state):
                # Random action for terminal state or 0
                actions = self.env.get_actions(state)
                new_policy[state] = actions[0] if actions else 0
                continue
            
            old_action = self.policy.get(state)
            
            # Find the best action
            action_values = []
            for action in self.env.get_actions(state):
                self.env.reset()
                if hasattr(self.env, 'current_pos'):
                    self.env.current_pos = state
                elif hasattr(self.env, 'state'):
                    self.env.state = state
                    
                next_state, reward, done, _ = self.env.step(action)
                action_value = reward + self.gamma * V[next_state]
                action_values.append((action, action_value))
            
            best_action = max(action_values, key=lambda x: x[1])[0]
            new_policy[state] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        return new_policy, policy_stable
    
    def train(self, episodes=None) -> Tuple[Dict, Dict]:
        """Trains the agent using Policy Iteration"""
        states = self.env.get_states()
        # Initialize policy randomly
        self.policy = {s: np.random.choice(self.env.get_actions(s)) for s in states}
        
        iteration = 0
        pbar = tqdm(desc="Policy Iteration")
        while True:
            # Policy Evaluation
            self.V = self.policy_evaluation(self.policy)
            
            # Policy Improvement
            self.policy, policy_stable = self.policy_improvement(self.V)
            
            iteration += 1
            pbar.update(1)
            if policy_stable:
                pbar.set_postfix({"status": "converged", "iters": iteration})
                pbar.close()
                break
        
        return self.policy, self.V
        
        return self.policy, self.V

    def act(self, state):
        return self.policy.get(state, np.random.choice(self.env.get_actions(state)))
