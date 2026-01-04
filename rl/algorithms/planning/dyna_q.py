import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
from tqdm import tqdm
from ..base_agent import BaseAgent
from ...policies.epsilon_greedy import EpsilonGreedyPolicy

class DynaQ(BaseAgent):
    """
    Dyna-Q Algorithm
    Combines Q-learning with planning (simulated experience)
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, n_planning_steps=5):
        super().__init__(env)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_greedy = EpsilonGreedyPolicy(epsilon)
        self.n_planning_steps = n_planning_steps
        
        self.Q = defaultdict(lambda: defaultdict(float))
        self.model = {}  # Model[s][a] = (r, s')
        self.policy = {}
    
    def q_learning_update(self, state, action, reward, next_state, done):
        """Q-learning update"""
        if done:
            target = reward
        else:
            max_q = max(self.Q[next_state].values()) if self.Q[next_state] else 0.0
            target = reward + self.gamma * max_q
        
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
    
    def planning(self):
        """Planning steps using the learned model"""
        if not self.model:
            return
        
        for _ in range(self.n_planning_steps):
            # Choose a random state-action pair previously observed
            state_list = list(self.model.keys())
            state = state_list[np.random.choice(len(state_list))]
            
            action_list = list(self.model[state].keys())
            action = action_list[np.random.choice(len(action_list))]
            
            reward, next_state = self.model[state][action]
            done = hasattr(self.env, 'is_terminal') and self.env.is_terminal(next_state)
            
            # Update Q with simulated experience
            self.q_learning_update(state, action, reward, next_state, done)
    
    def train(self, episodes=5000) -> Tuple[Dict, Dict]:
        """Trains the agent using Dyna-Q"""
        for episode_num in tqdm(range(episodes), desc="Dyna-Q"):
            state = self.env.reset()
            done = False
            
            while not done:
                actions = self.env.get_actions(state)
                q_values = [self.Q[state][a] for a in actions]
                action = self.epsilon_greedy.select_action(q_values, actions)
                
                next_state, reward, done, _ = self.env.step(action)
                
                # Q-learning (direct RL)
                self.q_learning_update(state, action, reward, next_state, done)
                
                # Update model
                if state not in self.model:
                    self.model[state] = {}
                self.model[state][action] = (reward, next_state)
                
                # Planning
                self.planning()
                
                state = next_state
        
        # Extract greedy policy
        
        # Extract greedy policy
        for state in self.Q:
            if self.Q[state]:
                self.policy[state] = max(self.Q[state].items(), key=lambda x: x[1])[0]
        
        return self.policy, dict(self.Q)

    def act(self, state):
        return self.policy.get(state, np.random.choice(self.env.get_actions(state)))
