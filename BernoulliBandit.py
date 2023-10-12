from cProfile import label
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

class BernoulliBandit:
    def __init__(self, means):
        self.means = means
        self.turns = 0
        self.rewards = {}
        self.times_used = {}
        self.total_reward = 0
        self.cumulative_reward = []
        self.cumulative_regret = 0
        self.cum_exp_reg = 0
        self.total_regrets = []
        self.immediate_regret = []
        self.expected_regrets = []
        self.prob_best_arm = []
    def K(self):
        return len(self.means)
    
    def pull(self, a):
        num = random.random()
        if num <= self.means[a]:
            self.turns += 1
            self.total_reward += 1
            self.cumulative_reward.append(self.total_reward)
            self.times_used[a] = self.times_used.get(a, 0) + 1
            self.rewards[a] = (self.rewards.get(a, 0)*(self.times_used[a]-1) + 1)/self.times_used[a]
            return self.rewards[a]
        else:
            self.turns += 1
            self.cumulative_reward.append(self.total_reward)
            self.times_used[a] = self.times_used.get(a, 0) + 1
            self.rewards[a] = (self.rewards.get(a, 0)*(self.times_used[a]-1) + 0)/self.times_used[a]
            return self.rewards[a]

    def regret(self, a):
        return (max(self.means) - self.means[a])
    def expected_regret(self, a):
        return max(self.rewards.values()) - self.rewards[a]