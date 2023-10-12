# Required libraries
from cProfile import label  # This import seems unnecessary as it's not used in the code.
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path  # This is also not used in the code.
from BernoulliBandit import BernoulliBandit as BB

def epsilon(Bandit, n, eps):
    """Implementation of the ε-first exploration strategy on a Bernoulli bandit environment.
    
    Args:
        Bandit (BB): An object representing the Bernoulli bandit environment. 
                     It should have methods like `K()`, `pull()`, `expected_regret()`, 
                     and member variables like `cum_exp_reg`, `expected_regrets`, 
                     `immediate_regret`, `cumulative_regret`, and `total_regrets`.
        n (int): Total number of rounds or pulls to perform.
        eps (int): The number of rounds dedicated to exploration.
    """
    
    arms = Bandit.K()  # Get the number of arms
    
    # Exploration phase: Pull each arm a number of times determined by eps.
    for t in range(arms * eps):
        Bandit.pull(t % arms)
        
        # Update cumulative and immediate regrets
        Bandit.cum_exp_reg += Bandit.expected_regret(t % arms)
        Bandit.expected_regrets.append(Bandit.cum_exp_reg)
        Bandit.immediate_regret.append(Bandit.regret(t % arms))
        Bandit.cumulative_regret += Bandit.regret(t % arms)
        Bandit.total_regrets.append(Bandit.cumulative_regret)
        
    # Find the arm with the highest reward so far
    i = 0
    for idx, val in Bandit.rewards.items():
        if val == max(Bandit.rewards.values()):
            i = idx
            
    # Exploitation phase: Keep pulling the best arm found during exploration
    for t in range(arms * eps + 1, n):
        Bandit.pull(i)
        
        # Update cumulative and immediate regrets
        Bandit.cum_exp_reg += Bandit.expected_regret(i)
        Bandit.expected_regrets.append(Bandit.cum_exp_reg)
        Bandit.immediate_regret.append(Bandit.regret(i))
        Bandit.cumulative_regret += Bandit.regret(i)
        Bandit.total_regrets.append(Bandit.cumulative_regret)

# Initialize a Bernoulli bandit with certain arm probabilities
Bandit_Epsilon = BB([0.5, 0.55, 0.6, 0.65, 0.7])

# Run the ε-first algorithm on the bandit
epsilon(Bandit_Epsilon, 100000, 7)

# Print the ratio of the total reward to the maximum possible reward
print(Bandit_Epsilon.total_reward / (max(Bandit_Epsilon.means) * 100000))

# Print the average rewards of all arms
print(Bandit_Epsilon.rewards.values())

# Print the number of times each arm has been pulled
print(Bandit_Epsilon.times_used.values())
