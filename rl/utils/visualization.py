import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(rewards, window=10, title="Learning Curve"):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
    if len(rewards) >= window:
        smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(rewards)), smooth_rewards, color='red', label='Moving Average')
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_v_table(v_table, grid_size, title="Value Function"):
    # Assumes v_table is a dict or array for a grid environment
    grid = np.zeros(grid_size)
    for s, v in v_table.items():
        row, col = s // grid_size[1], s % grid_size[1]
        grid[row, col] = v
    
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title(title)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            plt.text(j, i, f"{grid[i,j]:.2f}", ha='center', va='center', color='white')
    plt.show()
