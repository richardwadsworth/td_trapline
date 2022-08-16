'''
 Python script to run a grid search across the Actor Critic  TC(Lambda) search space

 Uses Ray Tune and MlFlow open source libraries to create grid search table and sample 
 the hyper-parameter space in parallel

 Result visible in MlFlow UI.  http://127.0.0.1:5000/. If MlFlow is not running, navigate 
 to the MlFlow mlruns parent directory and run mlflow ui. Refer to mlflow documentation for
 more information.
'''
from gym_utils import register_gym
register_gym()
from runner_utils import train_parallel
from mrp import *
from json import loads, dumps

'''
######################################################
# hyper-parameters for Actor Critic TD(lambda) model
#
# EDIT the Parameter set below and then run the script.
#
######################################################
'''
mrp_function = mrp_10_positive_array_ohashi() # or mrp_10_positive_array_ohashi()

episodes = [200] # range of total number of episodes in training run
steps = [100] # range of  episode lengths

gamma =  [0.7, 0.8, 0.9] # discount factor
alpha_actor = [0.7] # actor learning rate
alpha_critic = [0.3] # critic learning rate
eligibility_decay = [0.6, 0.7, 0.8] # eligibility trace decay

#softmax temperature annealing
epsilon_start = 1
epsilon_end = 0.2

# epsilon_annealing_stop_ratio : As a percentage of the total number of episodes.  e.g if episodes=200 
# and epsilon_annealing_stop_ratio=0.2, epsilon will anneal linearly from epsilon_start to epsilon_end
# linearly over the first  0.2*200 steps.
epsilon_annealing_stop_ratio = [0.2, 0.5] # range.  

respiration_reward = [-0.01] # -1/np.square(size) # -1/(steps+(steps*0.1)) # negative reward for moving 1 step in an episode
stationary_reward = [-0.01] # respiration_reward*2 # positive reward for moving, to discourage not moving
revisit_inactive_target_reward = [0.0] # negative reward for revisiting an inactive target (i.e. one that has already been visited)
change_in_orientation_reward = [0]#-stationary_reward*0.5 #negative reward if orientation changes


'''
######################################################
#
# END
#
# hyper-parameters for Actor Critic TD(lambda) model
######################################################
'''

MRP = loads(mrp_function)
experiment_name = MRP["name"] + '_gs' # gs for grid search

if __name__ == "__main__":

    train_parallel(10,
                    MRP["size"], 
                    dumps(MRP), 
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

