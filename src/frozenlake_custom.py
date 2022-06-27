import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(1, '/Users/richard/Documents/projects/py_learning/sussex/Dissertation/gym')

from plots import plot_performance, initialise_plots
from gym_utils import initialise_gym

from rl_td import train
from q_function import initialise_q, print_q, print_optimal_q_policy
from policies import GreedyPolicy



# parameters for sarsa(lambda)
episodes = 3000
STEPS = 50
gamma = 0.9
alpha = 0.05
eligibility_decay = 0.3
T = 0.1 #softmax temperature

# #equalateral triangle
size = 8
MDP = np.array([(50,2.0), (22,1.0)]) #markov decision chain including rewards for each target

#straightish line
# size = 19
# MDP = np.array([(62,1.0), (181,1.0), (300, 1.0)]) #markov decision chain including rewards for each target

env = initialise_gym(size, MDP, STEPS)

do_in_epsisode_plots=True

print("Action space = ", env.action_space)
print("Observation space = ", env.observation_space)

env.reset()
env.render()

# initialise the action state values
q = initialise_q(env)

plot_data = initialise_plots(env)

# train the algorithm
q, performance, ax = train(env, episodes, STEPS, eligibility_decay, alpha, gamma, T, q, plot_data, do_in_epsisode_plots)

# visual the algorithm's performance
plot_performance(episodes, STEPS, performance, plot_data)

# get the final performance value of the algorithm using a greedy policy
greedyPolicyAvgPerf = GreedyPolicy(env).average_performance(q)
print("Greedy policy SARSA performance =", greedyPolicyAvgPerf) 

# print the final action state values
print_q(env, q)

# print the optimal policy in human readable form
print_optimal_q_policy(env, q)

env.close()
plt.show()


