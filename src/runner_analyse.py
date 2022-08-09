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
    # run = mlflow.get_run("0b07230d28ed43aabe9f04aaebe1afbe") #best 6 medium after mdp refactor
    # run = mlflow.get_run("8c76ffb6fae54f4893adfdf7804c1b7a") #best 10 medium after mdp refactor

    # run = mlflow.get_run("c1954e74680641d6a0a4aed9110fd575") #best 6 medium after dynamic nest refactor
    # run = mlflow.get_run("e7b4f076dad248828dc574816f7417a9") #best 10 medium after dynamic nest refactor

    run = mlflow.get_run("a8ba9383841a49b481372e5a1ece3af5") #best 6 medium after using lowest softmax for perf test
    #run = mlflow.get_run("e9e589b3596f4b10a5af8fe6273c9497") #best 10 medium after using lowest softmax for perf test
    
    #run = mlflow.get_run("5e4293a925fd4c9bbd69df400bd1b97b") #best 6 medium after catchup
    
    
    config = run.data.params

    experiment_name = "analyse_" + run.info.run_id + "_" + config["experiment_name"]
    analyse = train_parallel_with_config(100,
                    experiment_name, 
                    config)
    1==1

