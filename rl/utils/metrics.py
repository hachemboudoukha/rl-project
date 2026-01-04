import numpy as np

def calculate_mean_reward(rewards, last_n=100):
    if not rewards:
        return 0.0
    return np.mean(rewards[-last_n:])

def calculate_success_rate(successes, last_n=100):
    if not successes:
        return 0.0
    return np.mean(successes[-last_n:])
