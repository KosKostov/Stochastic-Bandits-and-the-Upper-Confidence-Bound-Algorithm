import random
import matplotlib.pyplot as plt
import numpy as np
import math
from BernoulliBandit import BernoulliBandit as BB

def Ucb_Asimp(Bandit, n):
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
                tm = Bandit.turns
                ucb = val + math.sqrt((2*math.log(1 + (tm*math.log(tm)**2)))/(Bandit.times_used[idx]))
                if max_value < ucb:
                    i = idx
                    max_value = ucb
            Bandit.pull(i)
            Bandit.cum_exp_reg += Bandit.expected_regret(i)
            Bandit.expected_regrets.append(Bandit.cum_exp_reg)
            Bandit.immediate_regret.append(Bandit.regret(i))
            Bandit.cumulative_regret += Bandit.regret(i)
            Bandit.total_regrets.append(Bandit.cumulative_regret)
