#plot learning over n experiments, showing error bars
import numpy as np
import matplotlib.pyplot as plt

from mlflow_utils import get_experiment_runs_data

import seaborn as sns
sns.set_theme(style="whitegrid")

#experiment_name = "analyse_d7f2c7c777164160bd5ac6bbfefb0a71_10_medium_negative_array_ohashi_gs_1000_runs" #best 10 negative ohashi, 200 episodes, 1000 runs
experiment_name = "analyse_3cb6d9c1e8c646188668a059a9190d6c_10_medium_positive_array_ohashi_gs_1000_runs" #best 10 positive ohashi, 200 episodes, 1000 runs


data, plot_rate = get_experiment_runs_data(experiment_name) 

all_runs_in_experiment = data["performance"]
all_runs_done = data["done"]

total_num_samples = all_runs_in_experiment.shape[0]

# filter out all runs that did not find all targets
runs_completed_target_sequence = []
for i in range(total_num_samples):
    if all_runs_done[i]:
        runs_completed_target_sequence.append(all_runs_in_experiment[i])

runs_completed_target_sequence=np.array(runs_completed_target_sequence)

num_performance_samples_per_episode = runs_completed_target_sequence.shape[1]

def plot_errors(num_samples, num_performance_samples_per_episode, plot_rate, data):
    xs, ys, zs = np.zeros(num_performance_samples_per_episode), np.zeros(num_performance_samples_per_episode), np.zeros(num_performance_samples_per_episode)
    for i in range(num_performance_samples_per_episode):
        xs[i] = i * plot_rate
        ys[i] = np.mean(data[:,i]) #mean performance for this episode
        stdev=np.std(data[:,i])
        zs[i]=stdev/(num_samples**0.5)

    return xs, ys, zs
    

#get data for where agent returned to nest after visiting all targets
xs1, ys1, zs1 = plot_errors(runs_completed_target_sequence.shape[0], num_performance_samples_per_episode, plot_rate, runs_completed_target_sequence) 

#get data for all training runs
xs2, ys2, zs2 = plot_errors(all_runs_in_experiment.shape[0], num_performance_samples_per_episode, plot_rate, all_runs_in_experiment)

# plot the data

fig, ax = plt.subplots()
fig.suptitle("Mean Learning Performance per Episode")
ax.set_title(experiment_name, fontsize=10)

ax.errorbar(xs1, ys1,yerr=zs1, label='All Targets Found')
ax.errorbar(xs2, ys2,yerr=zs2, label='All runs', alpha=0.7)

ax.legend(loc='upper left')
ax.set_xlabel('Episode')
ax.set_ylabel('Mean Reward Using Softmax Policy')

plt.show()
