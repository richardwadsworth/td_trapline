import numpy as np
from gym_utils import register_gym

register_gym()

from runner_utils import train_parallel
from mdp import *


# parameters for sarsa(lambda)
size, MDP, experiment_name = get_large_positive_array()

episodes = [100]
steps = [100, 150]

gamma = {"lower":0.7, "upper":0.9, "q":0.1} # discount factor
alpha_actor = alpha_critic = [0.3, 0.5, 0.7] # actor learning rate, critic learning rate
eligibility_decay = {"lower":0.7, "upper":0.8, "q":0.1}# eligibility trace decay

#softmax temperature annealing
epsilon_start = 1
epsilon_end = 0.2
epsilon_annealing_stop_ratio = {"lower":0.3, "upper":0.9, "q":0.3}

respiration_reward = [-0.01, -0.005] # -1/np.square(size) # -1/(steps+(steps*0.1)) # negative reward for moving 1 step in an episode
stationary_reward = [-0.01, -0.005] # respiration_reward*2 # positive reward for moving, to discourage not moving
revisit_inactive_target_reward = [-0.1, -0.01, 0.0] # negative reward for revisiting an inactive target (i.e. one that has already been visited)
change_in_orientation_reward = [0]#-stationary_reward*0.5 #negative reward if orientation changes
    
if __name__ == "__main__":

    train_parallel(1,
                    size, 
                    MDP, 
                    experiment_name, 
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
                    epsilon_annealing_stop_ratio)

