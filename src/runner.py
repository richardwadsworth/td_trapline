import matplotlib.pyplot as plt
import numpy as np

from gym_utils import register_gym, initialise_gym

register_gym(True)

from plots import plot_performance, initialise_plots

from rl_td import train
from q_function import initialise_actor, initialise_critic, get_q_pretty_print, get_optimal_q_policy_pretty_print
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


# opposite corner 9
# size = 9
# MDP = np.array([(np.square(size)-1,1.0)]) #markov decision chain including rewards for each target

# large straight-ish line
# size = 19
# MDP = np.array([(map_coord_to_index(size, 6, 6),1.0), 
#                 (map_coord_to_index(size, 10, 10),1.0),
#                 (map_coord_to_index(size, 14, 14),1.0)
#                 ])


# # large positive array
# size = 11
# MDP = np.array([(map_coord_to_index(size, 1, 1),1.0), 
#                 (map_coord_to_index(size, 5, 0),1.0),
#                 (map_coord_to_index(size, 0, 5),1.0),
#                 (map_coord_to_index(size, 8, 3),1.0),
#                 (map_coord_to_index(size, 3, 8),1.0),
#                 (map_coord_to_index(size, 7, 7),1.0)
#                 ])


# small positive array
size = 7
MDP = np.array([(map_coord_to_index(size, 1, 1),1.0), 
                (map_coord_to_index(size, 3, 1),1.0),
                (map_coord_to_index(size, 1, 3),1.0),
                (map_coord_to_index(size, 5, 3),1.0),
                (map_coord_to_index(size, 3, 5),1.0),
                (map_coord_to_index(size, 5, 5),1.0)
                ])


# # curved line
# size = 19
# MDP = np.array([(62,1.0), (198,1.0), (300, 1.0)]) #markov decision chain including rewards for each target

rng = np.random.default_rng() # random number generator

is_stochastic = False
plot_rate = 5 # rate at which to plot predictions

episodes = 100
steps = 200
gamma = 0.85 # discount factor
alpha_actor = 0.7 # actor learning rate, critic learning rate
alpha_critic = 0.5 # 
eligibility_decay = 0.1 # eligibility trace decay

#softmax temperature annealing
epsilon_start = 1
epsilon_end = 0.2
epsilon_annealing_stop_ratio = 0.7

respiration_reward = -0.01 # -1/np.square(size) # -1/(steps+(steps*0.1)) # negative reward for moving 1 step in an episode
stationary_reward = -0.01 # respiration_reward*2 # positive reward for moving, to discourage not moving
revisit_inactive_target_reward = -0.1 # negative reward for revisiting an inactive target (i.e. one that has already been visited)
change_in_orientation_reward = 0#-stationary_reward*0.5 #negative reward if orientation changes

env = initialise_gym(size, MDP, is_stochastic, respiration_reward, stationary_reward, revisit_inactive_target_reward, change_in_orientation_reward, steps)

record_stats = True
do_in_episode_plots=False
plot_data = None

print("Action space = ", env.action_space)
print("Observation space = ", env.observation_space)

if __name__ == "__main__":
    while True:

        env.reset()
        # env.render()

        # initialise the action state values
        actor = initialise_actor(env)
        critic = initialise_critic(env, rng)

        if do_in_episode_plots:
            plot_data = initialise_plots(env)

        policy_train = SoftmaxDirectionalPolicy(env, rng)
        policy_predict = GreedyDirectionalPolicy(env)

        import tempfile
        import os
        from tempfile import gettempdir
        artifact_dir = "./sussex/Dissertation/output"
        artifact_filepath = os.path.join(artifact_dir, '{}.dat'.format(hash(os.times())))
        if not os.path.exists(artifact_dir):
            os.mkdir(artifact_dir) 
        
        # train the algorithm
        actor, performance = train(env, 
            episodes, 
            steps, 
            eligibility_decay, 
            alpha_actor,
            alpha_critic, 
            gamma, 
            epsilon_start, 
            epsilon_end, 
            epsilon_annealing_stop_ratio, 
            actor,
            critic, 
            policy_train,
            policy_predict,
            plot_rate,
            plot_data,
            artifact_filepath,
            do_in_episode_plots,
            record_stats)

        print("Training performance mean: {}".format(np.mean(performance)))
        print("Training performance stdev: {}".format(np.std(performance)))

        if do_in_episode_plots:
            # visual the algorithm's performance
            plot_performance(episodes, steps, performance, plot_rate, plot_data)

        # get the final performance value of the algorithm using a greedy policy
        greedyPolicyAvgPerf =policy_predict.average_performance(policy_predict.action, q=actor)

        #get average action state values across all possible actions.  i.e. get a 2d slice of the 3d matrix
        q_mean = np.mean(actor, axis=(0))

        # print the final action state values
        print(get_q_pretty_print(env, q_mean))

        # print the optimal policy in human readable form
        print(get_optimal_q_policy_pretty_print(env, q_mean))

        print("Greedy policy SARSA performance =", greedyPolicyAvgPerf) 

        plt.show()
        env.close()

        if greedyPolicyAvgPerf > 5:
            print("Output file is " + artifact_filepath)
            print("End")
            print()
            break
        else:
            os.remove(artifact_filepath)



