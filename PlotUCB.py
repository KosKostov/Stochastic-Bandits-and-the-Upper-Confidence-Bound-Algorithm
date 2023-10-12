from cProfile import label
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path
plt.rcParams['svg.fonttype'] = 'none'
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
from BernoulliBandit import BernoulliBandit as BB
from UCB_Normal import Ucb_Generic as UCBG
from UCB_Asimpt_Best import Ucb_Asimp as UCBA

def UCBA(Bandit, n):
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
            

Bandit_Asimp = BB([0.4, 0.45, 0.5, 0.55, 0.6])
Bandit_Generic = BB([0.4, 0.45, 0.5, 0.55, 0.6])
Bandit_Epsilon = BB([0.4, 0.45, 0.5, 0.55, 0.6])
Bandit_MOSS = BB([0.4, 0.45, 0.5, 0.55, 0.6])

UCBA(Bandit_Asimp, 100000)
UCBG(Bandit_Generic, 100000)
epsilon(Bandit_Epsilon, 100000, 20)
Ucb_MOSS(Bandit_MOSS, 100000)

print(Bandit_Asimp.total_reward)
print(Bandit_Generic.total_reward)
print(Bandit_Epsilon.total_reward)
print(Bandit_MOSS.total_reward)
print(Bandit_Asimp.rewards.values())
print(Bandit_Generic.rewards.values())
print(Bandit_Epsilon.rewards.values())
print(Bandit_MOSS.rewards.values())
print(Bandit_Asimp.times_used.values())
print(Bandit_Generic.times_used.values())
print(Bandit_Epsilon.times_used.values())
print(Bandit_MOSS.times_used.values())






fig1 = plt.figure(figsize=(6,5), dpi=100)
axes1 = fig1.add_axes([0.12, 0.12, 0.82, 0.82])
axes1.plot(range(len(Bandit_Asimp.total_regrets)), Bandit_Asimp.total_regrets, label = "UCB_Asmp_Optimal")
axes1.plot(range(len(Bandit_Generic.total_regrets)), Bandit_Generic.total_regrets, label = "UCB_Normal")
axes1.plot(range(len(Bandit_Epsilon.total_regrets)), Bandit_Epsilon.total_regrets, label = "Explore_Commit")
axes1.set_title("The Accumulated Regret")
axes1.set_xlabel("Horizon")
axes1.set_ylabel('Regret')

axes1.plot(range(len(Bandit_MOSS.total_regrets)), Bandit_MOSS.total_regrets, label = "MOSS")
axes1.legend(loc=0)

"""fig2 = plt.figure(figsize=(5,4), dpi=100)
axes2 = fig2.add_axes([0.1, 0.1, 0.9, 0.9])
axes2.plot(range(len(Bandit_Asimp.expected_regrets)), Bandit_Asimp.expected_regrets, label = "Asmp_Optimal")
axes2.plot(range(len(Bandit_Generic.expected_regrets)), Bandit_Generic.expected_regrets, label = "Generic")
axes2.plot(range(len(Bandit_Epsilon.expected_regrets)), Bandit_Epsilon.expected_regrets, label = "Explore_Commit")
#axes2.plot(range(len(Bandit_MOSS.expected_regrets)), Bandit_MOSS.expected_regrets, label = "MOSS")
axes2.legend(loc=0)"""

fig3 = plt.figure(figsize=(6,5), dpi=100)
axes3 = fig3.add_axes([0.12, 0.12, 0.82, 0.82])
axes3.plot(range(len(Bandit_Asimp.cumulative_reward)), Bandit_Asimp.cumulative_reward, label = "UCB_Asmp_Optimal")
axes3.plot(range(len(Bandit_Generic.cumulative_reward)), Bandit_Generic.cumulative_reward, label = "UCB_Normal")
axes3.plot(range(len(Bandit_Epsilon.cumulative_reward)), Bandit_Epsilon.cumulative_reward, label = "Explore_Commit")
axes3.set_title("The Accumulated Reward")
axes3.set_xlabel("Horizon")
axes3.set_ylabel("Reward")
axes3.plot(range(len(Bandit_MOSS.cumulative_reward)), Bandit_MOSS.cumulative_reward, label = "MOSS")
axes3.legend(loc=0)
plt.show()



