'''
script for running one set of hyper parameters in massive parallel for further analysis
'''
import numpy as np
from gym_utils import register_gym

register_gym()

from runner_utils import train_parallel_with_config
import mlflow


if __name__ == "__main__":

    #run = mlflow.get_run("2859cc9d8c3242918c9af22cdcb6b5d9")
    run = mlflow.get_run("0b07230d28ed43aabe9f04aaebe1afbe") #best 6 medium after mdp refactor
    run = mlflow.get_run("8c76ffb6fae54f4893adfdf7804c1b7a") #best 10 medium after mdp refactor

    
    
    
    config = run.data.params

    experiment_name = "analyse_" + run.info.run_id + "_" + config["experiment_name"]
    analyse = train_parallel_with_config(100,
                    experiment_name, 
                    config)
    1==1

