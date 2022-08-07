#plot learning over n experiments, showing error bars
import mlflow
import numpy as np
import matplotlib.pyplot as plt

experiment_path = ""
record_stats = True
sim_data = []

experiment = mlflow.get_experiment_by_name("analyse_bc764671207f4bf5b14a2f445083d0c6_10_medium_positive_array_offset")
experiment = mlflow.get_experiment_by_name("analyse_2859cc9d8c3242918c9xaf22cdcb6b5d9_6_medium_positive_array_offset")
print("Experiment name {}.".format(experiment.name))
experiment_artifact_location = experiment.artifact_location

import os
from urllib.parse import urlparse
p = urlparse(experiment_artifact_location)
files = [f for f in os.listdir(p.path) if not f=="meta.yaml"] # get a list of all files in the dir
runs_performance = []
runs_done = []

import tempfile
from mlflow.tracking import MlflowClient
        
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
            performance = result["performance"]
            done = result["done"]
            plot_rate = 5 #result["plot_rate"]
            runs_performance.append(performance)
            runs_done.append(done)


runs_performance = np.array(runs_performance)
total_num_samples = runs_performance.shape[0]

runs_returned_to_nest = []
for i in range(total_num_samples):
    if runs_done[i]:
        runs_returned_to_nest.append(runs_performance[i])

runs_returned_to_nest=np.array(runs_returned_to_nest)

num_performance_samples_per_episode = runs_returned_to_nest.shape[1]

def plot_errors(num_samples, num_performance_samples_per_episode, plot_rate, data):
    xs, ys, zs = np.zeros(num_performance_samples_per_episode), np.zeros(num_performance_samples_per_episode), np.zeros(num_performance_samples_per_episode)
    for i in range(num_performance_samples_per_episode):
        xs[i] = i * plot_rate
        ys[i] = np.mean(data[:,i]) #mean performance for this episode
        stdev=np.std(data[:,i])
        zs[i]=stdev/(num_samples**0.5)

    return xs, ys, zs
    

#get data for where agent returned to nest after visiting all targets
xs1, ys1, zs1 = plot_errors(runs_returned_to_nest.shape[0], num_performance_samples_per_episode, plot_rate, runs_returned_to_nest) 

#get data for all training runs
xs2, ys2, zs2 = plot_errors(runs_performance.shape[0], num_performance_samples_per_episode, plot_rate, runs_performance)

# plot the data
plt.errorbar(xs1, ys1,yerr=zs1, label='Returned to nest')
plt.errorbar(xs2, ys2,yerr=zs2, label='All runs', alpha=0.7)

plt.legend(loc='upper left')
plt.xlabel('Episode')
plt.ylabel('Mean Softmax Performance')
plt.title('Average Learning Performance per Episode')
plt.show()
