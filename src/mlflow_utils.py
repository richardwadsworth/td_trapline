#plot learning over n experiments, showing error bars
import mlflow
import numpy as np
import os
from urllib.parse import urlparse
import tempfile
from mlflow.tracking import MlflowClient
import json

def get_experiment_runs_data(experiment_name):

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment == None:
        raise ValueError("MlFlow experiment '{}' not found.".format(experiment_name))
    print("Experiment name {}.".format(experiment.name))

    experiment_artifact_location = experiment.artifact_location

    p = urlparse(experiment_artifact_location)
    files = [f for f in os.listdir(p.path) if not f=="meta.yaml"] # get a list of all files in the dir
    all_observations = []
    all_performances = []
    all_runs = []
    all_metrics = []
    all_params = []
    all_run_ids = []

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
                all_run_ids.append(run.info.run_id)
                all_params.append(run.data.params)
                all_metrics.append(run.data.metrics)
                all_observations.append(observations)
                all_performances.append(performance)
                all_runs.append(done)
    
    MRP = json.loads(run.data.params["MRP"]) # capture the MRP used the last run,  which will be the same for all runs in this experiment
            
    data = {"observations": np.array(all_observations,dtype=object),
            "performance": np.array(all_performances),
            "done": np.array(all_runs),
            "MRP": MRP,
            "metrics" : np.array(all_metrics,dtype=object),
            "params" : np.array(all_params,dtype=object),
            "run_ids" : np.array(all_run_ids , dtype=str)
        }


    return data