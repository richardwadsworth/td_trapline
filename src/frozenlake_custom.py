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
from q_function import initialise_q, print_q, print_optimal_q_policy
from policies import GreedyPolicy

# %%
# parameters for sarsa(lambda)
episodes = 500
STEPS = 100
gamma = 0.9
alpha = 0.05
eligibility_decay = 0.3
tau = 1 #softmax temperature

env = initialise_gym(STEPS)

# %%
print("Action space = ", env.action_space)
print("Observation space = ", env.observation_space)

env.reset()
env.render()

# initialise the action state values
q = initialise_q(env)

# train the algorithm
q, performance = train(env, episodes, STEPS, eligibility_decay, alpha, gamma, tau, q)

# visual the algorithm's performance
plot_performance(episodes, STEPS, performance)

# get the final performance value of the algorithm using a greedy policy
greedyPolicyAvgPerf = GreedyPolicy(env).average_performance(q)
print("Greedy policy SARSA performance =", greedyPolicyAvgPerf) 

# print the action state values
print_q(env, q)


print_optimal_q_policy(env, q)
# %%
def resolveActionDict(x, actionState):
    if x in env.goal_indices:
        return " ! "
    elif all(v == 0 for v in actionState):
        return " - "
    else:
        return actionsDict[np.argmax(actionState)]
    

# %%


# print("Optimal policy:")
# idxs = [0,4,8,12]
# for idx in idxs:
#     print(optimalPolicy[idx+0], optimalPolicy[idx+1], 
#           optimalPolicy[idx+2], optimalPolicy[idx+3])

# %%

env.close()
plt.show()


