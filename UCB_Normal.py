import random
import matplotlib.pyplot as plt
import numpy as np
import math
from BernoulliBandit import BernoulliBandit as BB

class BernoulliBandit_Generic:
    def __init__(self, means):
        self.means = means
        self.turns = 0
        self.rewards = {}
        self.times_used = {}
        self.total_reward = 0
        self.cumulative_reward = []
        self.im_regrets = []
        self.total_regret = []
        self.expected_regret = []
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

    def regret(self):
        return (max(self.rewards.values())*self.turns - self.total_reward)
    def exp_regret(self, a):
        return max(self.rewards.values()) - self.rewards[a]
    def immediate_regret(self, a):
        im_reg = max(self.rewards.values()) - self.means[a]
        self.im_regrets.append(im_reg)
        return
def Ucb_Generic(Bandit, n):
    for t in range(n):
        arms = Bandit.K()
        if t < arms:
            Bandit.pull(t)
            Bandit.cum_exp_reg += Bandit.expected_regret(t)
            Bandit.expected_regrets.append(Bandit.cum_exp_reg)
            Bandit.immediate_regret.append(Bandit.regret(t))
            Bandit.cumulative_regret += Bandit.regret(t)
            Bandit.total_regrets.append(Bandit.cumulative_regret)
            
        else:
            max_value = 0
            i = 0
            for idx, val in Bandit.rewards.items():
                ucb = val + math.sqrt((2*math.log(n**2))/(Bandit.times_used[idx]))
                if max_value < ucb:
                    i = idx
                    max_value = ucb
            Bandit.pull(i)
            Bandit.cum_exp_reg += Bandit.expected_regret(i)
            Bandit.expected_regrets.append(Bandit.cum_exp_reg)
            Bandit.immediate_regret.append(Bandit.regret(i))
            Bandit.cumulative_regret += Bandit.regret(i)
            Bandit.total_regrets.append(Bandit.cumulative_regret)
