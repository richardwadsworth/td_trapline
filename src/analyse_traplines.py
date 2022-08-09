#plot learning over n experiments, showing error bars

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from mlflow_utils import get_experiment_runs_data
from utils import get_sliding_window_sequence
from trapline import get_optimal_trapline_for_diamond_array, is_stable_trapline, get_trapline_for_run_using_route_distribution, get_ordered_target_list_for_episode

#data, plot_rate = get_experiment_runs_data("analyse_bc764671207f4bf5b14a2f445083d0c6_10_medium_positive_array_offset")
#data, plot_rate = get_experiment_runs_data("analyse_2859cc9d8c3242918c9af22cdcb6b5d9_6_medium_positive_array_offset")

#data, plot_rate = get_experiment_runs_data("analyse_0b07230d28ed43aabe9f04aaebe1afbe_6_medium_positive_array_offset") #after MDP refactor
#data, plot_rate = get_experiment_runs_data("analyse_8c76ffb6fae54f4893adfdf7804c1b7a_10_medium_positive_array_offset") #after MDP refactor

#data, plot_rate = get_experiment_runs_data("analyse_c1954e74680641d6a0a4aed9110fd575_6_medium_positive_array_offset") #best 6 medium after dynamic nest refactor
#data, plot_rate = get_experiment_runs_data("analyse_e7b4f076dad248828dc574816f7417a9_10_medium_positive_array_offset") #best 10 medium after dynamic nest refactor

# data, plot_rate = get_experiment_runs_data("analyse_5e4293a925fd4c9bbd69df400bd1b97b_6_medium_positive_array_offset") #best 10 medium after perftest use min softmax
# #data, plot_rate = get_experiment_runs_data("analyse_e9e589b3596f4b10a5af8fe6273c9497_10_medium_positive_array_offset") #best 10 medium after perftest use min softmax

# all_run_sample_episodes_in_experiment = data["observations"]
# all_run_sample_done_in_experiment = data["done"]
# MDP = data["MDP"]


# pickle.dump( all_run_sample_episodes_in_experiment, open( "all_run_sample_episodes_in_experiment.p", "wb" ) )
# pickle.dump( all_run_sample_done_in_experiment, open( "all_run_sample_done_in_experiment.p", "wb" ) )
# pickle.dump( MDP, open( "all_MDP.p", "wb" ) )


all_run_sample_episodes_in_experiment = pickle.load( open( "all_run_sample_episodes_in_experiment.p", "rb" ) )
all_run_sample_done_in_experiment = pickle.load( open( "all_run_sample_done_in_experiment.p", "rb" ) )
MDP = pickle.load( open( "all_MDP.p", "rb" ) )
MDP["size"] = 12

num_runs_in_experiment = all_run_sample_episodes_in_experiment.shape[0]
num_sample_episodes_per_run = all_run_sample_episodes_in_experiment.shape[1]

def get_route_index_from_observation(route_observations):
    route_indexes = [[observation[0] for observation in observations] for observations in route_observations]
    return route_indexes



optimal_trapline_master, optimal_trapline_reversed_master = get_optimal_trapline_for_diamond_array(MDP["targets"])

# initalise results arrays
results = pd.DataFrame()
results["route"] = np.array(num_runs_in_experiment, dtype=object)
results["count"] = np.zeros(num_runs_in_experiment, dtype=int)
results["stable"] = np.zeros(num_runs_in_experiment, dtype=bool)
            
STABLE_POINT = 0.25 # the ratio of latter sample episodes to use in this algorithm to determine the trapline

# get the sliding sequence used to determine what the trapline route is
sliding_sequence_used_for_identifying_trapline_route = get_sliding_window_sequence(5, num_sample_episodes_per_run, STABLE_POINT)
# get the sliding widow to use in determining if there is a stable trapline
sliding_sequence_used_for_trapline_stability = get_sliding_window_sequence(2, num_sample_episodes_per_run, STABLE_POINT)


#for each episode, find the order that the the targets where discovered in
for run_index in range(num_runs_in_experiment):

    run_episodes_route_observations = all_run_sample_episodes_in_experiment[run_index] # all the observations in a specific run.  i,e. all the episodes and their runs

    run_episodes_routes = get_route_index_from_observation(run_episodes_route_observations) #extract the route indexes from the route observations
    
    run_episodes_targets_found = np.zeros((num_sample_episodes_per_run), dtype=object) # array of an ordered list of the order targets were discovered for each sample episode in this run
    
    #filter out all steps that are not targets.  only take the first time the target was hit (i.e. ignore when a target is revisited)
    for episode_index in range(num_sample_episodes_per_run):
    
        route = run_episodes_routes[episode_index]
        
        run_episodes_targets_found[episode_index] = get_ordered_target_list_for_episode(optimal_trapline_master, MDP["nest"], route)

    route, count = get_trapline_for_run_using_route_distribution(sliding_sequence_used_for_identifying_trapline_route, run_episodes_targets_found)

    # use the sliding sequence to determine is there is a stable trapline
    stable_trapline_found = is_stable_trapline(MDP["size"], sliding_sequence_used_for_trapline_stability, run_episodes_routes, 100)

    
    results.loc[run_index, 'route'] = str(route if stable_trapline_found else []) #only save the route if a stable trapline was found
    results.loc[run_index, 'count'] = count
    results.loc[run_index, 'stable'] = stable_trapline_found
    
# get a count of all the different routes of the traplines from each run
count3 = pd.Series(results["route"]).value_counts().sort_values(ascending=False)

LABEL_NO_TRAPLINE_FOUND = 'No trapline found'
x = [str(x if x!='[]' else LABEL_NO_TRAPLINE_FOUND) for x in count3.index.to_list()] # convert index (which is the route), to a str
fig, ax = plt.subplots(1,1)
barlist = ax.bar(x, count3) # plot the bar chart


# add nest to optimal route
optimal_trapline_master.append(MDP["nest"])
optimal_trapline_master_inc_nest = str(optimal_trapline_master)

# add nest to reverse of optimal route
optimal_trapline_reversed_master.append(MDP["nest"])
optimal_trapline_reversed_master_inc_nest = str(optimal_trapline_reversed_master)

# highlight the optimal traplines if present in the results
for l in range(len(x)):
    index = x[l]
    if index == LABEL_NO_TRAPLINE_FOUND:
        barlist[l].set_color('r')
    elif (index == optimal_trapline_master_inc_nest) or (index == optimal_trapline_reversed_master_inc_nest):
        barlist[l].set_color('g')


ax.set_xticklabels(x, rotation = 90)
fig.tight_layout()
plt.grid()
plt.show()