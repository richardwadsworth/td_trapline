'''
 Python script to run the td learning model using a specific arena and parameter set without mlflow.  
 
 Both sets of variable are set in the script.

 The script will repeat in a loop until the cumulative reward of the sumulation is greater than

 Run this script using the command line python ./src/runner.py
'''
from gym_utils import register_gym
register_gym(True) # note we are running in DEVELOPMENT MODE here and the code will use the gym src in ./modules NOT the gym installed by your package manager!

import numpy as np
from runner_utils import train_fnn
from plots import PlotType
from mrp import *
from json import loads
'''
###########################################
 Parameters for td learning model

 Start
 ###########################################
'''
# 




MRP = mrp_10_negative_array_ohashi()
threshold = 7.3# the cumulative reward needed to stop the simulation loop kicked off by this script
do_in_episode_plots=PlotType.Full # set the plot frequency

rng = np.random.default_rng() # random number generator

episodes = 200
steps = 100
gamma = 0.8 # discount factor
alpha_actor = 0.7 # actor learning rate, critic learning rate
alpha_critic = 0.3 # 
eligibility_decay = 0.8 # eligibility trace decay

#softmax temperature annealing
epsilon_start = 1
epsilon_end = 0.2
epsilon_annealing_stop_ratio = 0.2

respiration_reward = -0.05 # -1/np.square(size) # -1/(steps+(steps*0.1)) # negative reward for moving 1 step in an episode
stationary_reward = -0.01 # respiration_reward*2 # positive reward for moving, to discourage not moving
revisit_inactive_target_reward = -0.0 # negative reward for revisiting an inactive target (i.e. one that has already been visited)
change_in_orientation_reward = 0#-stationary_reward*0.5 #negative reward if orientation changes


'''
###########################################
 Parameters for td learning model

 End
 ###########################################
'''

is_stochastic = False
plot_rate = 5 # rate at which to plot predictions
record_stats = True


size = int(loads(MRP)["size"])

def main():

    train_fnn(is_stochastic,
            size, 
            MRP, 
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
            rng,
            threshold=threshold)

if __name__ == "__main__":
   main()


