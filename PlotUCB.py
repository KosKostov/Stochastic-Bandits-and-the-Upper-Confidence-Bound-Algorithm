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
from UCB_MOSS import Ucb_MOSS as UCBM
from epsilon import epsilon as naive_approach
            
Bandit_Asimp = BB([0.4, 0.45, 0.5, 0.55, 0.6])
Bandit_Generic = BB([0.4, 0.45, 0.5, 0.55, 0.6])
Bandit_Epsilon = BB([0.4, 0.45, 0.5, 0.55, 0.6])
Bandit_MOSS = BB([0.4, 0.45, 0.5, 0.55, 0.6])

UCBA(Bandit_Asimp, 100000)
UCBG(Bandit_Generic, 100000)
naive_approach(Bandit_Epsilon, 100000, 20)
UCBM(Bandit_MOSS, 100000)

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



