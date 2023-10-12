# Required Libraries
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from BernoulliBandit import BernoulliBandit as BB

def Ucb_Asimp(Bandit, n):
    """Implementation of the Asymptotic Upper Confidence Bound (UCB) algorithm on a Bernoulli bandit environment.
    
    Args:
        Bandit (BB): An object representing the Bernoulli bandit environment. This object should have methods like 
                     `K()`, `pull()`, `expected_regret()`, and member variables like `cum_exp_reg`, `expected_regrets`, 
                     `immediate_regret`, `cumulative_regret`, and `total_regrets`.
        n (int): Total number of rounds or pulls to perform.
    """
    
    for t in range(n):
        arms = Bandit.K()  # Get the number of arms
        
        if t < arms:
            # If we haven't pulled each arm once, do it now
            Bandit.pull(t)
            
            # Update cumulative and immediate regrets
            Bandit.cum_exp_reg += Bandit.expected_regret(t)
            Bandit.expected_regrets.append(Bandit.cum_exp_reg)
            Bandit.immediate_regret.append(Bandit.regret(t))
            Bandit.cumulative_regret += Bandit.regret(t)
            Bandit.total_regrets.append(Bandit.cumulative_regret)
            
        else:
            # If we have pulled each arm at least once, calculate the UCB values to decide the next arm to pull
            max_value = 0
            i = 0
            tm = Bandit.turns  # Total number of turns taken so far
            
            # Iterate through all the arms to calculate UCB values with an asymptotic adjustment
            for idx, val in Bandit.rewards.items():
                ucb = val + math.sqrt((2 * math.log(1 + (tm * math.log(tm)**2))) / (Bandit.times_used[idx]))
                
                if max_value < ucb:
                    i = idx
                    max_value = ucb
            
            # Pull the arm with the highest UCB value
            Bandit.pull(i)
            
            # Update cumulative and immediate regrets
            Bandit.cum_exp_reg += Bandit.expected_regret(i)
            Bandit.expected_regrets.append(Bandit.cum_exp_reg)
            Bandit.immediate_regret.append(Bandit.regret(i))
            Bandit.cumulative_regret += Bandit.regret(i)
            Bandit.total_regrets.append(Bandit.cumulative_regret)
