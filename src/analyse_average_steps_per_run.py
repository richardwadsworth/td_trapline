"""
 Python script to plot the average number of steps in each episode of a training run, across X number of samples

 Plot learning over n experiments, showing error bars

 Call this script passing the in the experiment name as the argument.

 e.g. 
 python .src/analyse_average_steps_per_run.py 382a_mrp_10_negative_array_ohashi_1661026904124893_16610278550079429
 
"""


import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from mlflow_utils import get_experiment_runs_data
import seaborn as sns
sns.set_theme(style="whitegrid")

def main():

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    args = parser.parse_args()
    experiment_name = args.experiment_name

    data = get_experiment_runs_data(experiment_name) 

    all_runs_in_experiment = data["observations"]
    all_runs_done = data["done"]
    sample_rate = int(data["params"][0]["plot_rate"])

    artifact_path = "artifacts"

    total_num_samples = all_runs_in_experiment.shape[0]

    # pre process observations.  extract the index position out of each step taken for each episode
    runs_routes = []
    for i in range(total_num_samples):
        run_observations = all_runs_in_experiment[i] # all the observations in a specific run.  i,e. all the episodes and their runs

        # get the total number of steps taken for each episode in this run
        run_step_count_per_episode = [len(observations) for observations in run_observations]
        runs_routes.append(run_step_count_per_episode) # save the total number of steps
        
    runs_routes = np.array(runs_routes) # convert to array for easier processing

    # filter out all runs that did not find all targets
    runs_completed_target_sequence = []
    for i in range(total_num_samples):
        if all_runs_done[i]:
            runs_completed_target_sequence.append(runs_routes[i])

    runs_completed_target_sequence=np.array(runs_completed_target_sequence)

    num_samples_per_episode = runs_completed_target_sequence.shape[1]

    def calculate_errors(num_samples, num_samples_per_episode, sample_rate, data):
        """
        calculate standard error of the mean across all samples

        Args:
            num_samples: total number of samples in batch
            num_performance_samples_per_episode: number of samples in an episode
            sample_rate: how frequently the simulation data was sampled in a n episode
            data: the samples

        Returns:
            xs: array of episode ids
            ys: the performance data
            zs: the standard error of the mean
        """
        xs, ys, zs = np.zeros(num_samples_per_episode), np.zeros(num_samples_per_episode), np.zeros(num_samples_per_episode)
        for i in range(num_samples_per_episode):
            xs[i] = i * sample_rate
            ys[i] = np.mean(data[:,i]) #mean performance for this episode
            stdev=np.std(data[:,i])
            zs[i]=stdev/(num_samples**0.5)

        return xs, ys, zs
        

    #get data for where agent returned to nest after visiting all targets
    xs1, ys1, zs1 = calculate_errors(runs_completed_target_sequence.shape[0], num_samples_per_episode, sample_rate, runs_completed_target_sequence) 

    #get data for all training runs
    xs2, ys2, zs2 = calculate_errors(runs_routes.shape[0], num_samples_per_episode, sample_rate, runs_routes)

    # plot the data
    fig, ax = plt.subplots()
    fig.suptitle('Mean Num Steps per Episode')
    ax.set_title(experiment_name, fontsize=10)

    # ax.errorbar(xs1, ys1,yerr=zs1, label='All targets found') # uncomment this to see only valid runs
    ax.errorbar(xs2, ys2,yerr=zs2, label='All runs', alpha=0.7)

    ax.legend(loc='upper right')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Number of steps') 

    ax.set_ylim(20, 165)

    filepath = os.path.join(artifact_path, args.experiment_name + '_mean_steps.png')
    plt.savefig(filepath, bbox_inches='tight', format='png')

    plt.show()

if __name__ == "__main__":
   main()
