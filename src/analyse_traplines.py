#plot learning over n experiments, showing error bars

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from mlflow_utils import get_experiment_runs_data
from utils import get_sliding_window_sequence
from trapline import get_optimal_trapline_for_diamond_array, is_stable_trapline_1, is_stable_trapline_2, get_trapline_for_run_using_route_distribution, get_ordered_target_list_for_episode
from plots import plot_route

#data, plot_rate = get_experiment_runs_data("analyse_bc764671207f4bf5b14a2f445083d0c6_10_medium_positive_array_offset")
#data, plot_rate = get_experiment_runs_data("analyse_2859cc9d8c3242918c9af22cdcb6b5d9_6_medium_positive_array_offset")

#data, plot_rate = get_experiment_runs_data("analyse_0b07230d28ed43aabe9f04aaebe1afbe_6_medium_positive_array_offset") #after MDP refactor
#data, plot_rate = get_experiment_runs_data("analyse_8c76ffb6fae54f4893adfdf7804c1b7a_10_medium_positive_array_offset") #after MDP refactor

#data, plot_rate = get_experiment_runs_data("analyse_c1954e74680641d6a0a4aed9110fd575_6_medium_positive_array_offset") #best 6 medium after dynamic nest refactor
#data, plot_rate = get_experiment_runs_data("analyse_e7b4f076dad248828dc574816f7417a9_10_medium_positive_array_offset") #best 10 medium after dynamic nest refactor

#data, plot_rate = get_experiment_runs_data("analyse_5e4293a925fd4c9bbd69df400bd1b97b_6_medium_positive_array_offset") #best 10 medium after perftest use min softmax
#data, plot_rate = get_experiment_runs_data("analyse_e9e589b3596f4b10a5af8fe6273c9497_10_medium_positive_array_offset") #best 10 medium after perftest use min softmax

#data, plot_rate = get_experiment_runs_data("analyse_ee9e2444031644129f8414dae1540094_get_10_medium_negative_array_chittka") #best 10 negative after manhattan, 200 episodes
data, plot_rate = get_experiment_runs_data("analyse_d1e93bc2a1654c649f49ce2e31b103eb_get_10_medium_negative_array_chittka") #best 10 negative after manhattan, 250 episodes
all_run_sample_episodes_in_experiment = data["observations"]
all_run_sample_done_in_experiment = data["done"]
MDP = data["MDP"]


# pickle.dump( all_run_sample_episodes_in_experiment, open( "all_run_sample_episodes_in_experiment.p", "wb" ) )
# pickle.dump( all_run_sample_done_in_experiment, open( "all_run_sample_done_in_experiment.p", "wb" ) )
# pickle.dump( MDP, open( "all_MDP.p", "wb" ) )


# all_run_sample_episodes_in_experiment = pickle.load( open( "all_run_sample_episodes_in_experiment.p", "rb" ) )
# all_run_sample_done_in_experiment = pickle.load( open( "all_run_sample_done_in_experiment.p", "rb" ) )
# MDP = pickle.load( open( "all_MDP.p", "rb" ) )
# MDP["size"] = 16

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
SLIDING_WINDOW_SIZE_USED_FOR_IDENTIFYING_TRAPLINE_ROUTE =5
SLIDING_WINDOW_SIZE_USED_FOR_TRAPLINE_STABILITY = 2
# get the sliding sequence used to determine what the trapline route is
sliding_sequence_used_for_identifying_trapline_route = get_sliding_window_sequence(SLIDING_WINDOW_SIZE_USED_FOR_IDENTIFYING_TRAPLINE_ROUTE, num_sample_episodes_per_run, STABLE_POINT)
# get the sliding widow to use in determining if there is a stable trapline
sliding_sequence_used_for_trapline_stability = get_sliding_window_sequence(SLIDING_WINDOW_SIZE_USED_FOR_TRAPLINE_STABILITY, num_sample_episodes_per_run, STABLE_POINT)


#for each episode, find the order that the the targets where discovered in
for run_index in range(num_runs_in_experiment):

    run_episodes_route_observations = all_run_sample_episodes_in_experiment[run_index] # all the observations in a specific run.  i,e. all the episodes and their runs

    run_episodes_routes = get_route_index_from_observation(run_episodes_route_observations) #extract the route indexes from the route observations
    
    run_episodes_targets_found = np.zeros((num_sample_episodes_per_run), dtype=object) # array of an ordered list of the order targets were discovered for each sample episode in this run
    
    #filter out all steps that are not targets.  only take the first time the target was hit (i.e. ignore when a target is revisited)
    for episode_index in range(num_sample_episodes_per_run):
    
        route = run_episodes_routes[episode_index]
        
        run_episodes_targets_found[episode_index] = get_ordered_target_list_for_episode(optimal_trapline_master, MDP["nest"], route)

    route, route_count_for_run = get_trapline_for_run_using_route_distribution(sliding_sequence_used_for_identifying_trapline_route, run_episodes_targets_found)

    # use the sliding sequence to determine is there is a stable trapline
    # stable_trapline_found = is_stable_trapline_1(MDP["size"], sliding_sequence_used_for_trapline_stability, run_episodes_routes, 100) #version1
    stable_trapline_found = is_stable_trapline_2(MDP["size"], sliding_sequence_used_for_trapline_stability, run_episodes_routes, 300) #version2 using segmentation
    
    results.loc[run_index, 'route'] = str(route if stable_trapline_found else []) #only save the route if a stable trapline was found
    results.loc[run_index, 'count'] = route_count_for_run
    results.loc[run_index, 'stable'] = stable_trapline_found
    
# get a count of all the different routes of the traplines from each run
route_count_for_experiment = pd.Series(results["route"]).value_counts().sort_values(ascending=False)
#print(route_count_for_experiment)

LABEL_NO_ROUTE_FOUND = 'No route found'

# reformat data frame for plotting
df = pd.DataFrame(route_count_for_experiment)
df['count'] = df['route']
df['route'] = df.index.to_list()
df.index = np.arange(0, len(df))

# build x-axis labels
counter = 1
x_axis = []
for i, r in enumerate(df['route']):
    if r == '[]':
        x_axis.append(LABEL_NO_ROUTE_FOUND)
    else:
        x_axis.append(str(counter))
        counter += 1
df['x-axis'] = x_axis


#plot bar chart
fig1, ax = plt.subplots(1,1)
bar_list = ax.bar(df['x-axis'], df['count']) # plot the bar chart
ax.set_yscale('log') # use logarithmic scale
ax.set_xlabel('Routes')
ax.set_ylabel('Logarithmic Count of Routes')

# add nest to optimal route
optimal_trapline = optimal_trapline_master.copy()
optimal_trapline.append(MDP["nest"])
optimal_trapline_inc_nest = str(optimal_trapline)

# add nest to reverse of optimal route
optimal_trapline_reversed = optimal_trapline_reversed_master.copy()
optimal_trapline_reversed.append(MDP["nest"])
optimal_trapline_reversed_inc_nest = str(optimal_trapline_reversed)

# highlight the optimal traplines if present in the results
for i in range(len(df)):
    label = df['x-axis'][i]
    route = df['route'][i]
    if label == LABEL_NO_ROUTE_FOUND:
        bar_list[i].set_color('r')
    elif (route == optimal_trapline_inc_nest) or (route == optimal_trapline_reversed_inc_nest):
        bar_list[i].set_color('g')

# ax.set_xticklabels(x, rotation = 90)
fig1.tight_layout()
plt.grid()

# drop the route count with no discernable target based route found
df = df.drop(df[df['x-axis'] == LABEL_NO_ROUTE_FOUND].index)

from json import loads
plot_size = int(np.ceil(np.sqrt(len(df))))
fig2, axs = plt.subplots(plot_size, plot_size, figsize=(plot_size*4, plot_size*4))

axs = np.array(axs).reshape(-1)

for i, ax in enumerate(axs):

    if i >= len(df):
        break
    # Convert string representation of route to list using json
    route = loads(df['route'][df.index[i]])

    plot_route(fig2, ax, MDP["size"],MDP["nest"], optimal_trapline_master, route, str(label))

plt.show()