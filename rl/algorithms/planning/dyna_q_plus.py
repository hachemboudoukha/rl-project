import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
from tqdm import tqdm
from ..base_agent import BaseAgent
from ...policies.epsilon_greedy import EpsilonGreedyPolicy

class DynaQPlus(BaseAgent):
    """
    Dyna-Q+ Algorithm
    Extension of Dyna-Q that encourages exploration of states not visited for a long time
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, 
                 n_planning_steps=5, kappa=0.001):
        super().__init__(env)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_greedy = EpsilonGreedyPolicy(epsilon)
        self.n_planning_steps = n_planning_steps
        self.kappa = kappa  # Exploration bonus
        
        self.Q = defaultdict(lambda: defaultdict(float))
        self.model = {}  # Model[s][a] = (reward, next_state)
        self.last_visit = defaultdict(lambda: defaultdict(int))  # Time since last visit
        self.time_step = 0
        self.policy = {}
    
    def q_learning_update(self, state, action, reward, next_state, done, bonus=0):
        """Q-learning update with exploration bonus"""
        if done:
            target = reward + bonus
        else:
            max_q = max(self.Q[next_state].values()) if self.Q[next_state] else 0.0
            target = reward + bonus + self.gamma * max_q
        
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
    
    def planning(self):
        """Planning steps with exploration bonus"""
        if not self.model:
            return
        
        for _ in range(self.n_planning_steps):
            state_list = list(self.model.keys())
            state = state_list[np.random.choice(len(state_list))]
            
            action_list = list(self.model[state].keys())
            action = action_list[np.random.choice(len(action_list))]
            
            reward, next_state = self.model[state][action]
            done = hasattr(self.env, 'is_terminal') and self.env.is_terminal(next_state)
            
            # Calculate exploration bonus
            tau = self.time_step - self.last_visit[state][action]
            bonus = self.kappa * np.sqrt(tau)
            
            self.q_learning_update(state, action, reward, next_state, done, bonus)
    
    def train(self, episodes=5000) -> Tuple[Dict, Dict]:
        """Trains the agent using Dyna-Q+"""
        for episode_num in tqdm(range(episodes), desc="Dyna-Q+"):
            state = self.env.reset()
            done = False
            
            while not done:
                actions = self.env.get_actions(state)
                q_values = [self.Q[state][a] for a in actions]
                action = self.epsilon_greedy.select_action(q_values, actions)
                
                next_state, reward, done, _ = self.env.step(action)
                
                # Q-learning (direct RL, no bonus for real experience)
                self.q_learning_update(state, action, reward, next_state, done, bonus=0)
                
                # Update model and visit time
                if state not in self.model:
                    self.model[state] = {}
                self.model[state][action] = (reward, next_state)
                self.last_visit[state][action] = self.time_step
                
                # Planning
                self.planning()
                
                state = next_state
                self.time_step += 1
        
        # Extract greedy policy
        
        # Extract greedy policy
        for state in self.Q:
            if self.Q[state]:
                self.policy[state] = max(self.Q[state].items(), key=lambda x: x[1])[0]
        
        return self.policy, dict(self.Q)

    def act(self, state):
        return self.policy.get(state, np.random.choice(self.env.get_actions(state)))
