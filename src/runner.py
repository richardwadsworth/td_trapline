import matplotlib.pyplot as plt
import numpy as np
import sys
import timeit

sys.path.insert(1, './sussex/Dissertation/gym')

from gym_utils import register_gym, initialise_gym

register_gym()

from plots import plot_performance, initialise_plots

from rl_td import train
from q_function import initialise_actor, initialise_critic, print_q, print_optimal_q_policy
from policies import GreedyDirectionalPolicy
from policies import SoftmaxDirectionalPolicy, SoftmaxFlattenedPolicy

# parameters for sarsa(lambda)

# opposite corner 4
# size = 4
# MDP = np.array([(np.square(size)-1,1.0)]) #markov decision chain including rewards for each target

#
# # opposite corner 13
# size = 13
# MDP = np.array([(np.square(size)-1,1.0)]) #markov decision chain including rewards for each target

# # opposite corner 19
# size = 19
# MDP = np.array([(np.square(size)-1,1.0)]) #markov decision chain including rewards for each target

# # equilateral triangle
# size = 8
# MDP = np.array([(50,1.0), (22,1.0)]) #markov decision chain including rewards for each target

def map_coord_to_index(size, x, y):
    return (x*size)+y

# hexagon
# size = 12
# MDP = np.array([(map_coord_to_index(size, 3, 3),1.0), 
#                 (map_coord_to_index(size, 2, 6),1.0),
#                 (map_coord_to_index(size, 6, 2),1.0),
#                 (map_coord_to_index(size, 5, 9),1.0),
#                 (map_coord_to_index(size, 9, 5),1.0),
#                 (map_coord_to_index(size, 8, 8),1.0)
#                 ])

# opposite corner 9
# size = 9
# MDP = np.array([(np.square(size)-1,1.0)]) #markov decision chain including rewards for each target

# large straight-ish line
# size = 19
# MDP = np.array([(map_coord_to_index(size, 6, 6),1.0), 
#                 (map_coord_to_index(size, 10, 10),1.0),
#                 (map_coord_to_index(size, 14, 14),1.0)
#                 ])


# large straight-ish line
size = 5
MDP = np.array([(map_coord_to_index(size, 1, 1),1.0), 
                (map_coord_to_index(size, 3, 3),1.0)
                ])


# # curved line
# size = 19
# MDP = np.array([(62,1.0), (198,1.0), (300, 1.0)]) #markov decision chain including rewards for each target

rng = np.random.default_rng() # random number generator

is_stochastic = False

episodes = 100
STEPS = 20 
plot_rate = 5 # rate at which to plot predictions
gamma = 0.9 # discount factor
alpha_actor = alpha_critic = 0.3 # actor learning rate, critic learning rate
# alpha_critic = 0.3 # 
eligibility_decay = 0.3 # eligibility trace decay

#softmax temperature annealing
epsilon_start = 1
epsilon_end = 0.2
epsilon_annealing_stop = int(episodes*0.75)

respiration_reward = -0.01 # -1/np.square(size) # -1/(STEPS+(STEPS*0.1)) # negative reward for moving 1 step in an episode
stationary_reward = -0.01 # respiration_reward*2 # positive reward for moving, to discourage not moving
revisit_inactive_target_reward = -0.1 # negative reward for revisiting an inactive target (i.e. one that has already been visited)
change_in_orientation_reward = 0#-stationary_reward*0.5 #negative reward if orientation changes

env = initialise_gym(size, MDP, is_stochastic, respiration_reward, stationary_reward, revisit_inactive_target_reward, change_in_orientation_reward, STEPS)

do_in_epsisode_plots=True

print("Action space = ", env.action_space)
print("Observation space = ", env.observation_space)

env.reset()
# env.render()

# initialise the action state values
actor = initialise_actor(env)
critic = initialise_critic(env, rng)

plot_data = initialise_plots(env)

policy_train = SoftmaxDirectionalPolicy(env, rng)
policy_predict = GreedyDirectionalPolicy(env)

# train the algorithm
actor, performance, ax = train(env, 
    episodes, 
    STEPS, 
    eligibility_decay, 
    alpha_actor,
    alpha_critic, 
    gamma, 
    epsilon_start, 
    epsilon_end, 
    epsilon_annealing_stop, 
    actor,
    critic, 
    policy_train,
    policy_predict,
    plot_rate,
    plot_data, 
    do_in_epsisode_plots, 
    rng)

print("Training performance mean: {}".format(np.mean(performance)))
print("Training performance stdev: {}".format(np.std(performance)))

# visual the algorithm's performance
plot_performance(episodes, STEPS, performance, plot_rate, plot_data)

# get the final performance value of the algorithm using a greedy policy
greedyPolicyAvgPerf =policy_predict.average_performance(policy_predict.action, q=actor)

#get average action state values across all possible actions.  i.e. get a 2d slice of the 3d matrix
q_mean = np.mean(actor, axis=(0))

# # print the final action state values
# print_q(env, q_mean)

# # print the optimal policy in human readable form
# print_optimal_q_policy(env, q_mean)

print("Greedy policy SARSA performance =", greedyPolicyAvgPerf) 

plt.show()
env.close()



