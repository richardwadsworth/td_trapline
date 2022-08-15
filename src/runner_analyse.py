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
    
    run_id = "3cb6d9c1e8c646188668a059a9190d6c" # positive 10
    #run_id = "" # negative 10
    
    run = mlflow.get_run(run_id) #best 10 positive, 250 episodes

    config = run.data.params

    experiment_name = "analyse_" + run.info.run_id + "_" + config["experiment_name"] + '_' + str(NUM_REPEAT_RUNS) + '_runs'
    analyse = train_parallel_with_config(NUM_REPEAT_RUNS,
                    experiment_name, 
                    config)
    1==1

