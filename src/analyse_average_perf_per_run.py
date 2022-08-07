#plot learning over n experiments, showing error bars
import numpy as np
import matplotlib.pyplot as plt

from mlflow_utils import get_experiment_runs_data

#data, plot_rate = get_experiment_runs_data("analyse_bc764671207f4bf5b14a2f445083d0c6_10_medium_positive_array_offset")
data, plot_rate = get_experiment_runs_data("analyse_2859cc9d8c3242918c9af22cdcb6b5d9_6_medium_positive_array_offset")

all_runs_in_experiment = data["performance"]
all_runs_done = data["done"]

total_num_samples = all_runs_in_experiment.shape[0]

# filter out all runs that did not return to the nest
runs_returned_to_nest = []
for i in range(total_num_samples):
    if all_runs_done[i]:
        runs_returned_to_nest.append(all_runs_in_experiment[i])

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
xs2, ys2, zs2 = plot_errors(all_runs_in_experiment.shape[0], num_performance_samples_per_episode, plot_rate, all_runs_in_experiment)

# plot the data
plt.errorbar(xs1, ys1,yerr=zs1, label='Returned to nest')
plt.errorbar(xs2, ys2,yerr=zs2, label='All runs', alpha=0.7)

plt.legend(loc='upper left')
plt.xlabel('Episode')
plt.ylabel('Mean Softmax Performance')
plt.title('Average Learning Performance per Episode')
plt.show()
