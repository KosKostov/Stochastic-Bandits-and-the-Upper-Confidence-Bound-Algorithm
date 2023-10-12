import random
import matplotlib.pyplot as plt
import numpy as np
import math
from BernoulliBandit import BernoulliBandit as BB

def log_plus(x):
    if x > 1:
        return math.log(x)
    else:
        return 0
def Ucb_MOSS(Bandit, n):
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
                ucb = val + math.sqrt((4/Bandit.times_used[idx])*log_plus(n/(5*Bandit.times_used[idx])))
                if max_value < ucb:
                    i = idx
                    max_value = ucb
            Bandit.pull(i)
            Bandit.cum_exp_reg += Bandit.expected_regret(i)
            Bandit.expected_regrets.append(Bandit.cum_exp_reg)
            Bandit.immediate_regret.append(Bandit.regret(i))
            Bandit.cumulative_regret += Bandit.regret(i)
            Bandit.total_regrets.append(Bandit.cumulative_regret)
Bandit1 = BB([0.9, 0.7, 0.5, 0.4, 0.3])
        
Ucb_MOSS(Bandit1, 10000)

#print(Bandit1.times_used.keys())
#print(Bandit1.times_used.values())
#print(Bandit1.rewards.keys())
#print(Bandit1.regrets)
#print(Bandit1.rewards.values())
#print(Bandit1.total_reward)
#print(Bandit1.regret())
x = range(len(Bandit1.total_regrets))
plt.plot(x, Bandit1.total_regrets)
plt.show()