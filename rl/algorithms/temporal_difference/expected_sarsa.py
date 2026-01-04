import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
from tqdm import tqdm
from ..base_agent import BaseAgent
from ...policies.epsilon_greedy import EpsilonGreedyPolicy

class ExpectedSARSA(BaseAgent):
    """
    Expected SARSA
    Uses expected value over next actions instead of sample
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(env)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_helper = EpsilonGreedyPolicy(epsilon)
        self.Q = defaultdict(lambda: defaultdict(float))
        self.policy = {}
    
    def expected_q_value(self, state):
        """Calculates expected Q value under epsilon-greedy policy"""
        if not self.Q[state]:
            return 0.0
        
        actions = self.env.get_actions(state)
        n_actions = len(actions)
        
        # Find the best action
        best_action = max(self.Q[state].items(), key=lambda x: x[1])[0]
        
        # Calculate expectation
        expected_value = 0.0
        for action in actions:
            if action == best_action:
                prob = (1 - self.epsilon) + self.epsilon / n_actions
            else:
                prob = self.epsilon / n_actions
            expected_value += prob * self.Q[state][action]
        
        return expected_value
    
    def train(self, episodes=5000) -> Tuple[Dict, Dict]:
        """Trains the agent using Expected SARSA"""
        for episode_num in tqdm(range(episodes), desc="Expected SARSA"):
            state = self.env.reset()
            done = False
            
            while not done:
                actions = self.env.get_actions(state)
                q_values = [self.Q[state][a] for a in actions]
                action = self.policy_helper.select_action(q_values, actions)
                
                next_state, reward, done, _ = self.env.step(action)
                
                # Expected SARSA update
                if done:
                    target = reward
                else:
                    expected_q = self.expected_q_value(next_state)
                    target = reward + self.gamma * expected_q
                
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
