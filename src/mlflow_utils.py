#plot learning over n experiments, showing error bars
import mlflow
import numpy as np
import os
from urllib.parse import urlparse
import tempfile
from mlflow.tracking import MlflowClient
import json

def get_experiment_runs_data(experiment_name, plot_rate=5):

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment == None:
        raise ValueError("MlFlow experiment not found. {}.".format(experiment_name))
    print("Experiment name {}.".format(experiment.name))

    experiment_artifact_location = experiment.artifact_location

    p = urlparse(experiment_artifact_location)
    files = [f for f in os.listdir(p.path) if not f=="meta.yaml"] # get a list of all files in the dir
    all_observations = []
    all_performances = []
    all_runs = []

    client = MlflowClient()
            
    with tempfile.TemporaryDirectory() as tmpdirname:
        for filename in files:

            #read file name
            run = mlflow.get_run(filename)
            
            # Download artifacts
            local_path = client.download_artifacts(run.info.run_id, "result.json", tmpdirname)
            
            from json import load
            with open(local_path, "r") as read_content:
                result = load(read_content)
                observations = result["observations"]
                performance = result["performance"]
                done = result["done"]
                plot_rate = plot_rate # TODO.  read this from the run's parameters
                all_observations.append(observations)
                all_performances.append(performance)
                all_runs.append(done)
    
    MDP = json.loads(run.data.params["MDP"]) # capture the MDP used the last run,  which will be the same for all runs in this experiment
            
    data = {"observations": np.array(all_observations,dtype=object),
            "performance": np.array(all_performances),
            "done": np.array(all_runs),
            "MDP": MDP
        }


    return data, plot_rate