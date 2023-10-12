# Required Libraries
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

# Define the BernoulliBandit class to simulate a multi-armed Bernoulli bandit environment.
class BernoulliBandit:
    def __init__(self, means):
        """Initialize the BernoulliBandit object with given arm means."""
        self.means = means  # List of means for each arm
        self.turns = 0  # Total number of pulls/turns taken on the bandit
        self.rewards = {}  # Store reward obtained for each arm over time
        self.times_used = {}  # Count how many times each arm was pulled
        self.total_reward = 0  # Total rewards obtained so far
        self.cumulative_reward = []  # List storing cumulative rewards over time
        self.cumulative_regret = 0  # Cumulative regret over time
        self.cum_exp_reg = 0  # Cumulative expected regret over time
        self.total_regrets = []  # List storing total regrets over time
        self.immediate_regret = []  # List storing immediate regret after each pull
        self.expected_regrets = []  # List storing expected regrets over time
        self.prob_best_arm = []  # List storing probability of choosing the best arm over time
    
    def K(self):
        """Return the number of arms in the bandit."""
        return len(self.means)
    
    def pull(self, a):
        """Simulate pulling arm 'a' and return the observed reward."""
        num = random.random()
        if num <= self.means[a]:
            # If random number is less than or equal to mean of arm 'a', give reward 1
            self._update_statistics(a, 1)
            return self.rewards[a]
        else:
            # Else, no reward is given
            self._update_statistics(a, 0)
            return self.rewards[a]
    
    def _update_statistics(self, a, reward):
        """Helper function to update statistics after each arm pull."""
        self.turns += 1
        self.total_reward += reward
        self.cumulative_reward.append(self.total_reward)
        self.times_used[a] = self.times_used.get(a, 0) + 1
        self.rewards[a] = (self.rewards.get(a, 0) * (self.times_used[a] - 1) + reward) / self.times_used[a]

    def regret(self, a):
        """Calculate and return the regret for pulling arm 'a'."""
        return (max(self.means) - self.means[a])

    def expected_regret(self, a):
        """Calculate and return the expected regret for pulling arm 'a'."""
        return max(self.rewards.values()) - self.rewards[a]
