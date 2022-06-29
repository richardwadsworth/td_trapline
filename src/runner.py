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

# # opposite corner 4
# size = 4
# MDP = np.array([(np.square(size)-1,1.0)]) #markov decision chain including rewards for each target

# # opposite corner 9
# size = 9
# MDP = np.array([(np.square(size)-1,1.0)]) #markov decision chain including rewards for each target

# # opposite corner 13
# size = 13
# MDP = np.array([(np.square(size)-1,1.0)]) #markov decision chain including rewards for each target

# # opposite corner 19
# size = 19
# MDP = np.array([(np.square(size)-1,1.0)]) #markov decision chain including rewards for each target

# # equilateral triangle
# size = 8
# MDP = np.array([(50,1.0), (22,1.0)]) #markov decision chain including rewards for each target

# straight-ish line
size = 19
MDP = np.array([(62,1.0), (181,1.0), (300, 1.0)]) #markov decision chain including rewards for each target

# # curved line
# size = 19
# MDP = np.array([(62,1.0), (198,1.0), (300, 1.0)]) #markov decision chain including rewards for each target

rng = np.random.default_rng() # random number generator

episodes = 2000
STEPS = 200
gamma = 0.9 # discount factor
alpha = 0.05 # learning rate
eligibility_decay = 0.3 # eligibility trace decay

#softmax temperature annealing
epsilon_start = 1
epsilon_end = 0.25
epsilon_annealing_stop = int(episodes*0.6)

respiration_reward =  -1/(STEPS+(STEPS*0.1)) # negative reward for moving 1 step in an episode
movement_reward = respiration_reward*2 # positive reward for moving, to discourage not moving
change_in_orientation_reward = -movement_reward*0.5 #negative reward if orientation changes

env = initialise_gym(size, MDP, respiration_reward, movement_reward, change_in_orientation_reward, STEPS)

do_in_epsisode_plots=True

print("Action space = ", env.action_space)
print("Observation space = ", env.observation_space)

env.reset()
env.render()

# initialise the action state values
q = initialise_q(env)

plot_data = initialise_plots(env)

# train the algorithm
q, performance, ax = train(env, 
    episodes, 
    STEPS, 
    eligibility_decay, 
    alpha, 
    gamma, 
    epsilon_start, 
    epsilon_end, 
    epsilon_annealing_stop, 
    q, 
    plot_data, 
    do_in_epsisode_plots, 
    rng)

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



