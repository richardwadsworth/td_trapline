# %%
import numpy as np

import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '/Users/richard/Documents/projects/py_learning/sussex/Dissertation/gym')

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

from plots import plotAgentPath, plotActionStateQuiver, plot_performance
from gym_utils import initialise_gym, getGoalCoordinates

from rl_td import train
from q_function import initialise_q
from policies import GreedyPolicy

# %%
# parameters for sarsa(lambda)
episodes = 500
STEPS = 100
gamma = 0.9
alpha = 0.05
# epsilon_start = 0.2
# epsilon_end = 0.001
# epsilon_annealing_stop = int(episodes/2)
eligibility_decay = 0.3
tau = 1 #softmax temperature


env = initialise_gym(STEPS)

# %%
print("Action space = ", env.action_space)
print("Observation space = ", env.observation_space)

# %%
actionsDict = {}
actionsDict[0] = " L "
actionsDict[1] = " D "
actionsDict[2] = " R "
actionsDict[3] = " U "

actionsDictInv = {}
actionsDictInv["L"] = 0
actionsDictInv["D"] = 1
actionsDictInv["R"] = 2
actionsDictInv["U"] = 3

# %%
env.reset()
env.render()

# %%
# optimalPolicy = ["R/D"," R "," D "," L ",
#                  " D "," - "," D "," - ",
#                  " R ","R/D"," D "," - ",
#                  " - "," R "," R "," ! ",]
    
# print("Optimal policy:")
# idxs = [0,4,8,12]
# for idx in idxs:
#     print(optimalPolicy[idx+0], optimalPolicy[idx+1], 
#           optimalPolicy[idx+2], optimalPolicy[idx+3])

# %%
# def action_epsilon_greedy(q, s, epsilon=0.05):
#     if np.random.rand() > epsilon:
#         return np.argmax(q[s])
#     return np.random.randint(4)

# def get_action_epsilon_greedy(epsilon):
#     return lambda q,s: action_epsilon_greedy(q, s, epsilon=epsilon)

# %%




q = initialise_q(env)

q, performance = train(env, episodes, STEPS, eligibility_decay, alpha, gamma, tau, q)


# %%
plot_performance(episodes, STEPS, performance)

# %%
greedyPolicyAvgPerf = GreedyPolicy(env).average_performance(q)
print("Greedy policy SARSA performance =", greedyPolicyAvgPerf) 

# %%
q = np.round(q,3)
print("(A,S) Value function =", q.shape)

for i, j in enumerate(np.arange(0, env.observation_space.n, env.size)):
    print("Row {}".format(i))    
    print(q[j:j+env.size,:])

# %%
def resolveActionDict(x, actionState):
    if x in env.goal_indices:
        return " ! "
    elif all(v == 0 for v in actionState):
        return " - "
    else:
        return actionsDict[np.argmax(actionState)]
    

# %%

policyFound = [resolveActionDict(x, q[x,:]) for x in range(env.observation_space.n)]

print("Greedy policy found:")
idxs = np.arange(0, env.observation_space.n, int(np.sqrt(env.observation_space.n)))
for idx in idxs:
    row = []
    for i in range(int(np.sqrt(env.observation_space.n))):
        row.append(policyFound[idx+i])
    print(','. join(row))
        
print(" ")

# print("Optimal policy:")
# idxs = [0,4,8,12]
# for idx in idxs:
#     print(optimalPolicy[idx+0], optimalPolicy[idx+1], 
#           optimalPolicy[idx+2], optimalPolicy[idx+3])

# %%

env.close()
plt.show()


