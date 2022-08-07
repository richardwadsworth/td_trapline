#plot learning over n experiments, showing error bars
import numpy as np
import matplotlib.pyplot as plt

from mlflow_utils import get_experiment_runs_data

#data, plot_rate = get_experiment_runs_data("analyse_bc764671207f4bf5b14a2f445083d0c6_10_medium_positive_array_offset")
#data, plot_rate = get_experiment_runs_data("analyse_2859cc9d8c3242918c9af22cdcb6b5d9_6_medium_positive_array_offset")

#data, plot_rate = get_experiment_runs_data("analyse_0b07230d28ed43aabe9f04aaebe1afbe_6_medium_positive_array_offset") #after MDP refactor
data, plot_rate = get_experiment_runs_data("analyse_8c76ffb6fae54f4893adfdf7804c1b7a_10_medium_positive_array_offset")

all_runs_in_experiment = data["observations"]
all_runs_done = data["done"]


total_num_samples = all_runs_in_experiment.shape[0]

# pre process observations.  extract the index position out of each step taken for each episode
runs_routes = []
for i in range(total_num_samples):
    run_observations = all_runs_in_experiment[i] # all the observations in a specific run.  i,e. all the episodes and their runs

    # get the total number of steps taken for each episode in this run
    run_step_count_per_episode = [len(observations) for observations in run_observations]
    runs_routes.append(run_step_count_per_episode) # save the total number of steps
    
runs_routes = np.array(runs_routes) # convert to array for easier processing


runs_returned_to_nest = []
for i in range(total_num_samples):
    if all_runs_done[i]:
        runs_returned_to_nest.append(runs_routes[i])

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
xs2, ys2, zs2 = plot_errors(runs_routes.shape[0], num_performance_samples_per_episode, plot_rate, runs_routes)

# plot the data
plt.errorbar(xs1, ys1,yerr=zs1, label='Returned to nest')
plt.errorbar(xs2, ys2,yerr=zs2, label='All runs', alpha=0.7)

plt.legend(loc='upper left')
plt.xlabel('Episode')
plt.ylabel('Number of steps')
plt.title('Average Num Steps per Episode')
plt.show()
