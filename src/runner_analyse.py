'''
script for running one set of hyper parameters in massive parallel for further analysis

 Call this script by passing the in the run id generated by analyse_hyperparameter_sweep.py as the argument.

 e.g. 
 python ./src/runner_analyse.py a628d5c7a59047629b7721ac09455aea
 
'''
from gym_utils import register_gym
register_gym()

import argparse
import numpy as np
from runner_utils import train_parallel_with_config
import mlflow
import time

def main():

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()

    NUM_REPEAT_RUNS = 1000
    
    run = mlflow.get_run(args.run_id) #best 10 positive, 250 episodes

    config = run.data.params

    uniq_filename_suffix = str(time.time() * 1000).replace('.','')
    experiment_name = str(run.info.run_id)[0:5] + "_" + config["experiment_name"] + '_' + uniq_filename_suffix

    analyse = train_parallel_with_config(NUM_REPEAT_RUNS,
                    experiment_name, 
                    config)

if __name__ == "__main__":
   main()