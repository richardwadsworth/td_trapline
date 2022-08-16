#plot learning over n experiments, showing error bars
import mlflow
import numpy as np
import os
from urllib.parse import urlparse
import tempfile
from mlflow.tracking import MlflowClient
import json

def get_experiment_runs_data(experiment_name):

    print("Loading data from mlflow.  This can take a few minutes ...")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment == None:
        raise ValueError("MlFlow experiment '{}' not found.".format(experiment_name))
    print("Experiment name {}.".format(experiment.name))

    client = MlflowClient()

    run_infos = client.list_run_infos(experiment.experiment_id)

    all_observations = []
    all_performances = []
    all_runs = []
    all_metrics = []
    all_params = []
    all_run_ids = []
        
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        for run_info in run_infos:

            #read file name
            run = mlflow.get_run(run_info.run_id)
            
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