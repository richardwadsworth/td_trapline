import matplotlib.pyplot as plt
import numpy as np
import sys
import timeit

sys.path.insert(1, '/Users/richard/Documents/projects/py_learning/sussex/Dissertation/gym')

from gym_utils import register_gym, initialise_gym

register_gym()

from plots import plot_performance, initialise_plots

from rl_td import train
from q_function import initialise_q, print_q, print_optimal_q_policy
from policies import GreedyDirectionalPolicy


# parameters for sarsa(lambda)
episodes = 3000
STEPS = 200
gamma = 0.9
alpha = 0.05
eligibility_decay = 0.3

#softmax temperature annealing
epsilon_start = 1
epsilon_end = 0.2
epsilon_annealing_stop = int(episodes/2)


# # opposite corner 9
# size = 9
# MDP = np.array([(80,1.0)]) #markov decision chain including rewards for each target

# opposite corner 9
size = 9
MDP = np.array([(80,1.0)]) #markov decision chain including rewards for each target


respiration_reward = -7/(STEPS+(STEPS*0.1)) # reward for moving 1 step in an episode
inactive_reward = -0.1 # reward for action resulting in no movement
orientation_reward_reduction_ratio = 0.99

# # oppostite corner 19
# size = 19
# MDP = np.array([(300,1.0)]) #markov decision chain including rewards for each target


# # #equalateral triangle
# size = 8
# MDP = np.array([(50,1.0), (22,1.0)]) #markov decision chain including rewards for each target

#straightish line
# size = 19
# MDP = np.array([(62,1.0), (181,1.0), (300, 1.0)]) #markov decision chain including rewards for each target


env = initialise_gym(size, MDP, respiration_reward, inactive_reward, orientation_reward_reduction_ratio, STEPS)

do_in_epsisode_plots=True

print("Action space = ", env.action_space)
print("Observation space = ", env.observation_space)

env.reset()
env.render()

# initialise the action state values
q = initialise_q(env)

plot_data = initialise_plots(env)

# train the algorithm
q, performance, ax = train(env, episodes, STEPS, eligibility_decay, alpha, gamma, epsilon_start, epsilon_end, epsilon_annealing_stop, q, plot_data, do_in_epsisode_plots)

# visual the algorithm's performance
plot_performance(episodes, STEPS, performance, plot_data)

# get the final performance value of the algorithm using a greedy policy
policy = GreedyDirectionalPolicy(env)

greedyPolicyAvgPerf =policy.average_performance(policy.action, q)

#get average action state values across all possible actions.  i.e. get a 2d slice of the 3d matrix
q_mean = np.mean(q, axis=(0))

# print the final action state values
print_q(env, q_mean)

# print the optimal policy in human readable form
print_optimal_q_policy(env, q_mean)

print("Greedy policy SARSA performance =", greedyPolicyAvgPerf) 

plt.show()
env.close()



