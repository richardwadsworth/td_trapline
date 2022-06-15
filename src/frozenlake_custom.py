import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '/Users/richard/Documents/projects/py_learning/sussex/Dissertation/gym')

from plots import plot_performance, initialise_plots
from gym_utils import initialise_gym

from rl_td import train
from q_function import initialise_q, print_q, print_optimal_q_policy
from policies import GreedyPolicy

# %%
# parameters for sarsa(lambda)
episodes = 30000
STEPS = 500
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

plot_data = initialise_plots(env)

# train the algorithm
q, performance, ax = train(env, episodes, STEPS, eligibility_decay, alpha, gamma, tau, q, plot_data, True)

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


