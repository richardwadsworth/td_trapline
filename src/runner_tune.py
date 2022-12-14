'''
 Python script to run a grid search across the Actor Critic  TC(Lambda) search space

 Uses Ray Tune and MlFlow open source libraries to create grid search table and sample 
 the hyper-parameter space in parallel

 Result visible in MlFlow UI.  http://127.0.0.1:5000/. If MlFlow is not running, navigate 
 to the MlFlow mlruns parent directory and run mlflow ui. Refer to mlflow documentation for
 more information.

 Call this script passing the in the mrp function name as the argument.

 e.g. 
 python ./src/runner_tune.py mrp_10_positive_array_ohashi
 or
 python ./src/runner_tune.py mrp_10_negative_array_ohashi

'''
from gym_utils import register_gym
register_gym()
import argparse
from runner_utils import train_parallel
from mrp import *
from json import loads, dumps
import time

def main():

    funcdict = {
        "mrp_10_positive_array_ohashi": mrp_10_positive_array_ohashi(),
        "mrp_10_negative_array_ohashi": mrp_10_negative_array_ohashi()
    }

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("mrp_function_name")
    args = parser.parse_args()
    
    mrp_function  = funcdict.get(args.mrp_function_name, None)
    if mrp_function == None:
        parser.error("Argument mrp_function_name {} is not in the function dictionary.".format(args.mrp_function_name))

    episodes = [200] # range of total number of episodes in training run
    steps = [150] # range of  episode lengths

    gamma =  [0.5, 0.7, 0.9] # discount factor
    alpha_actor = [0.3, 0.5, 0.7, 0.9] # actor learning rate
    alpha_critic = [0.3, 0.5, 0.7, 0.9] # critic learning rate
    eligibility_decay = [0.4, 0.6, 0.8] # eligibility trace decay

    #softmax temperature annealing
    epsilon_start = 1
    epsilon_end = 0.2

    # epsilon_annealing_stop_ratio : As a percentage of the total number of episodes.  e.g if episodes=200 
    # and epsilon_annealing_stop_ratio=0.2, epsilon will anneal linearly from epsilon_start to epsilon_end
    # linearly over the first  0.2*200 steps.
    epsilon_annealing_stop_ratio = [0.2, 0.5, 0.9] # range.  

    respiration_reward = [-0.05] # -1/np.square(size) # -1/(steps+(steps*0.1)) # negative reward for moving 1 step in an episode
    stationary_reward = [-0.01] # respiration_reward*2 # positive reward for moving, to discourage not moving
    revisit_inactive_target_reward = [0.0] # negative reward for revisiting an inactive target (i.e. one that has already been visited)
    change_in_orientation_reward = [0]#-stationary_reward*0.5 #negative reward if orientation changes


    '''
    ######################################################

    END

    hyper-parameters for Actor Critic TD(lambda) model
    ######################################################
    '''

    uniq_filename_suffix = str(time.time() * 1000).replace('.','')

    MRP = loads(mrp_function)

    experiment_name = MRP["name"] + '_' + uniq_filename_suffix # gs for grid search

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


if __name__ == "__main__":
   main()