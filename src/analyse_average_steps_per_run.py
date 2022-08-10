#plot learning over n experiments, showing error bars
import numpy as np
import matplotlib.pyplot as plt

from mlflow_utils import get_experiment_runs_data


#experiment_name = "analyse_32bed68ecebc40849485df2ad8d5958f_10_medium_positive_array_chittka" #best 10 positive chittka, 200 episodes
experiment_name = "analyse_dbe7b192cd70476dbd59e2e65153c1a5_10_medium_negative_array_chittka" #best 10 negative chittka, 200 episodes

data, plot_rate = get_experiment_runs_data(experiment_name) 


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

# filter out all runs that did not find all targets
runs_completed_target_sequence = []
for i in range(total_num_samples):
    if all_runs_done[i]:
        runs_completed_target_sequence.append(runs_routes[i])

runs_completed_target_sequence=np.array(runs_completed_target_sequence)

num_samples_per_episode = runs_completed_target_sequence.shape[1]

def plot_errors(num_samples, num_samples_per_episode, plot_rate, data):
    xs, ys, zs = np.zeros(num_samples_per_episode), np.zeros(num_samples_per_episode), np.zeros(num_samples_per_episode)
    for i in range(num_samples_per_episode):
        xs[i] = i * plot_rate
        ys[i] = np.mean(data[:,i]) #mean performance for this episode
        stdev=np.std(data[:,i])
        zs[i]=stdev/(num_samples**0.5)

    return xs, ys, zs
    

#get data for where agent returned to nest after visiting all targets
xs1, ys1, zs1 = plot_errors(runs_completed_target_sequence.shape[0], num_samples_per_episode, plot_rate, runs_completed_target_sequence) 

#get data for all training runs
xs2, ys2, zs2 = plot_errors(runs_routes.shape[0], num_samples_per_episode, plot_rate, runs_routes)

# plot the data
plt.errorbar(xs1, ys1,yerr=zs1, label='All targets found')
plt.errorbar(xs2, ys2,yerr=zs2, label='All runs', alpha=0.7)

plt.legend(loc='upper left')
plt.xlabel('Episode')
plt.ylabel('Number of steps')
plt.title('Average Num Steps per Episode')
plt.show()
