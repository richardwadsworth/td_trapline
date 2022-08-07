import numpy as np

from gym_utils import register_gym
register_gym(True)


from runner_utils import train_fnn
from plots import PlotType
from mdp import *

# parameters for sarsa(lambda)
size, MDP, _ = get_10_medium_positive_array()
rng = np.random.default_rng() # random number generator

episodes = 100
steps = 200
gamma = 0.8 # discount factor
alpha_actor = 0.7 # actor learning rate, critic learning rate
alpha_critic = 0.3 # 
eligibility_decay = 0.8 # eligibility trace decay

#softmax temperature annealing
epsilon_start = 1
epsilon_end = 0.2
epsilon_annealing_stop_ratio = 0.5

respiration_reward = -0.005 # -1/np.square(size) # -1/(steps+(steps*0.1)) # negative reward for moving 1 step in an episode
stationary_reward = -0.005 # respiration_reward*2 # positive reward for moving, to discourage not moving
revisit_inactive_target_reward = -0.0 # negative reward for revisiting an inactive target (i.e. one that has already been visited)
change_in_orientation_reward = 0#-stationary_reward*0.5 #negative reward if orientation changes

is_stochastic = False
plot_rate = 5 # rate at which to plot predictions
record_stats = True
do_in_episode_plots=PlotType.Partial # None,Minimal, Partial, Full

if __name__ == "__main__":

    train_fnn(is_stochastic,
            size, 
            MDP, 
            respiration_reward,
            stationary_reward,
            revisit_inactive_target_reward,
            change_in_orientation_reward,
            episodes,
            steps,
            eligibility_decay,
            alpha_actor,
            alpha_critic,
            gamma,
            epsilon_start,
            epsilon_end,
            epsilon_annealing_stop_ratio,
            plot_rate,
            do_in_episode_plots,
            record_stats,
            rng)

        


