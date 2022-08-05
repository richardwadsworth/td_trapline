import matplotlib.pyplot as plt
import numpy as np

from gym_utils import register_gym, initialise_gym

register_gym(True)

from plots import plot_performance, initialise_plots, plot_traffic_greyscale, plot_traffic_noise, PlotType
from rl_td import train
from q_function import initialise_actor, initialise_critic, get_q_pretty_print, get_optimal_q_policy_pretty_print
from policies import GreedyDirectionalPolicy
from policies import SoftmaxDirectionalPolicy, SoftmaxFlattenedPolicy
from mdp import *

# parameters for sarsa(lambda)
size, MDP, _ = get_large_positive_array()

rng = np.random.default_rng() # random number generator

is_stochastic = False
plot_rate = 5 # rate at which to plot predictions

episodes = 100
steps = 150
gamma = 0.8 # discount factor
alpha_actor = 0.7 # actor learning rate, critic learning rate
alpha_critic = 0.3 # 
eligibility_decay = 0.8 # eligibility trace decay

#softmax temperature annealing
epsilon_start = 1
epsilon_end = 0.2
epsilon_annealing_stop_ratio = 0.2

respiration_reward = -0.005 # -1/np.square(size) # -1/(steps+(steps*0.1)) # negative reward for moving 1 step in an episode
stationary_reward = -0.005 # respiration_reward*2 # positive reward for moving, to discourage not moving
revisit_inactive_target_reward = -0.0 # negative reward for revisiting an inactive target (i.e. one that has already been visited)
change_in_orientation_reward = 0#-stationary_reward*0.5 #negative reward if orientation changes

env = initialise_gym(size, MDP, is_stochastic, respiration_reward, stationary_reward, revisit_inactive_target_reward, change_in_orientation_reward, steps)


record_stats = True
do_in_episode_plots=PlotType.Minimal # None,Minimal, Partial, Full
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

        if do_in_episode_plots != PlotType.NoPlots:
            plot_data = initialise_plots(env)

        policy_train = SoftmaxDirectionalPolicy(env, rng, 50)
        policy_predict = GreedyDirectionalPolicy(env)

        sim_data = []
        
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
            sim_data,
            do_in_episode_plots,
            record_stats)

        
        # # get the final performance value of the algorithm using a greedy policy
        # greedyPolicyAvgPerf = policy_predict.average_performance(policy_predict.action, q=actor)
        # softmaxPolicyAvgPerf = policy_train.average_performance(policy_train.get_action(epsilon_end), q=actor)

        print("Training performance mean: {}".format(np.mean(performance)))
        print("Training performance stdev: {}".format(np.std(performance)))

        if do_in_episode_plots != PlotType.NoPlots:
            # visual the algorithm's performance
            plot_performance(episodes, steps, performance, plot_rate, plot_data)
        
    
        # print("Greedy policy SARSA performance =", greedyPolicyAvgPerf) 
        # print("Softmax policy SARSA performance =", softmaxPolicyAvgPerf) 

        last_x_mean = np.mean(performance[-3])
        print("Training performance last x mean: {}".format(last_x_mean))
        
        if last_x_mean > 5:

            #get average action state values across all possible actions.  i.e. get a 2d slice of the 3d matrix
            q_mean = np.mean(actor, axis=(0))

            print(get_q_pretty_print(env, q_mean)) # print the final action state values
            print(get_optimal_q_policy_pretty_print(env, q_mean)) # print the optimal policy in human readable form

            softmaxPolicyAvgPerf, obs_data = policy_train.average_performance_with_observations(policy_train.get_action(epsilon_end), q=actor)
            print("Final Softmax policy SARSA performance =", softmaxPolicyAvgPerf) 

            import os   
            from statistics_utils import save_stats         
            artifact_dir = "./sussex/Dissertation/output"
            filename = '{}'.format(hash(os.times()))
            stats_filepath = os.path.join(artifact_dir, filename + '.dat')
            if not os.path.exists(artifact_dir):
                os.mkdir(artifact_dir) 
            
            save_stats(stats_filepath, sim_data) # save this simulation's data to disk

            fig1, _, _, _, _, ax5, ax6, xs_coordinate_map, ys_coordinate_map, xs_target, ys_target = plot_data
            plot_traffic_noise(env, fig1, ax5, xs_coordinate_map, ys_coordinate_map, xs_target, ys_target, sim_data, "Training")
            plot_traffic_noise(env, fig1, ax6, xs_coordinate_map, ys_coordinate_map, xs_target, ys_target, obs_data, "Test")
            print("Output file is " + stats_filepath)
            print("End")
            print()

            if do_in_episode_plots != PlotType.NoPlots:
                fig_filepath = os.path.join(artifact_dir, filename)
                plt.savefig(fig_filepath)
                plt.show()

            break

        env.close()

        


