from cProfile import label
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path
from BernoulliBandit import BernoulliBandit as BB

def epsilon(Bandit, n, eps):
    arms = Bandit.K()
    for t in range(arms*eps):
        Bandit.pull(t%arms)
        Bandit.cum_exp_reg += Bandit.expected_regret(t%arms)
        Bandit.expected_regrets.append(Bandit.cum_exp_reg)
        Bandit.immediate_regret.append(Bandit.regret(t%arms))
        Bandit.cumulative_regret += Bandit.regret(t%arms)
        Bandit.total_regrets.append(Bandit.cumulative_regret)
        
        i = 0
        for idx, val in Bandit.rewards.items():
            if val == max(Bandit.rewards.values()):
                i = idx
            
    for t in range(arms*eps+1, n):
        Bandit.pull(i)
        Bandit.cum_exp_reg += Bandit.expected_regret(i)
        Bandit.expected_regrets.append(Bandit.cum_exp_reg)
        Bandit.immediate_regret.append(Bandit.regret(i))
        Bandit.cumulative_regret += Bandit.regret(i)
        Bandit.total_regrets.append(Bandit.cumulative_regret)

Bandit_Epsilon = BB([0.5, 0.55, 0.6, 0.65, 0.7])


epsilon(Bandit_Epsilon, 100000, 7)


print(Bandit_Epsilon.total_reward/(max(Bandit_Epsilon.means)*100000))

print(Bandit_Epsilon.rewards.values())

print(Bandit_Epsilon.times_used.values())
