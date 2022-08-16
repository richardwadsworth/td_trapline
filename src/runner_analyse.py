'''
script for running one set of hyper parameters in massive parallel for further analysis
'''
import numpy as np
from gym_utils import register_gym

register_gym()

from runner_utils import train_parallel_with_config
import mlflow


if __name__ == "__main__":

    NUM_REPEAT_RUNS = 1000
    
    '''
    ######################################################
     EDIT THE run_id value to change which hyper-parameter set is used in the simulations
    ######################################################
    '''
    #run_id = "cb41737061ea435e8b43abfddc0258cf" # positive 10
    run_id = "bc6e223900244462b7898b0b511a9a4b" # negative 10
    '''
    ######################################################
     END

     run_id
    ######################################################
    '''
    
    run = mlflow.get_run(run_id) #best 10 positive, 250 episodes

    config = run.data.params

    experiment_name = "analyse_" + run.info.run_id + "_" + config["experiment_name"] + '_' + str(NUM_REPEAT_RUNS) + '_runs'
    analyse = train_parallel_with_config(NUM_REPEAT_RUNS,
                    experiment_name, 
                    config)
    1==1

