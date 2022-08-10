#plot learning over n experiments, showing error bars

    #find C for all sample routes
    #smooth C using sliding window
    #plot
    #traverse graph until C < threshold
    #find modal target sequence from that point


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from json import loads

from mlflow_utils import get_experiment_runs_data
from utils import get_sliding_window_sequence
from trapline import get_optimal_trapline_for_diamond_array, get_routes_similarity, get_valid_target_sequence_from_route
from plots import plot_route

#data, plot_rate = get_experiment_runs_data("analyse_bc764671207f4bf5b14a2f445083d0c6_10_medium_positive_array_offset")
#data, plot_rate = get_experiment_runs_data("analyse_2859cc9d8c3242918c9af22cdcb6b5d9_6_medium_positive_array_offset")

#data, plot_rate = get_experiment_runs_data("analyse_0b07230d28ed43aabe9f04aaebe1afbe_6_medium_positive_array_offset") #after MDP refactor
#data, plot_rate = get_experiment_runs_data("analyse_8c76ffb6fae54f4893adfdf7804c1b7a_10_medium_positive_array_offset") #after MDP refactor

#data, plot_rate = get_experiment_runs_data("analyse_c1954e74680641d6a0a4aed9110fd575_6_medium_positive_array_offset") #best 6 medium after dynamic nest refactor
#data, plot_rate = get_experiment_runs_data("analyse_e7b4f076dad248828dc574816f7417a9_10_medium_positive_array_offset") #best 10 medium after dynamic nest refactor

#data, plot_rate = get_experiment_runs_data("analyse_5e4293a925fd4c9bbd69df400bd1b97b_6_medium_positive_array_offset") #best 10 medium after perftest use min softmax
#experiment_name = "analyse_e9e589b3596f4b10a5af8fe6273c9497_10_medium_positive_array_offset" #best 10 medium after perftest use min softmax

#experiment_name = "analyse_ee9e2444031644129f8414dae1540094_get_10_medium_negative_array_chittka" #best 10 negative after manhattan, 200 episodes
# experiment_name = "analyse_d1e93bc2a1654c649f49ce2e31b103eb_get_10_medium_negative_array_chittka"  #best 10 negative after manhattan, 250 episodes

experiment_name = "analyse_692e2276ec7d4dd59fbb23ad49b41ce8_10_medium_positive_array_chittka" #best 10 positive chittka, 250 episodes
#experiment_name = "analyse_d1e93bc2a1654c649f49ce2e31b103eb_get_10_medium_negative_array_chittka" #best 10 negative chittka, 250 episodes

# data, plot_rate = get_experiment_runs_data(experiment_name) 
# all_run_sample_episodes_in_experiment = data["observations"]
# all_run_sample_done_in_experiment = data["done"]
# MDP = data["MDP"]


# pickle.dump( all_run_sample_episodes_in_experiment, open( "all_run_sample_episodes_in_experiment.p", "wb" ) )
# pickle.dump( all_run_sample_done_in_experiment, open( "all_run_sample_done_in_experiment.p", "wb" ) )
# pickle.dump( MDP, open( "all_MDP.p", "wb" ) )

all_run_sample_episodes_in_experiment = pickle.load( open( "all_run_sample_episodes_in_experiment.p", "rb" ) )
all_run_sample_done_in_experiment = pickle.load( open( "all_run_sample_done_in_experiment.p", "rb" ) )
MDP = pickle.load( open( "all_MDP.p", "rb" ) )


def get_route_index_from_observation(route_observations):
    route_indexes = [[observation[0] for observation in observations] for observations in route_observations]
    return route_indexes

def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

def get_C_scores_index_for_run(size, sliding_window_sequence, routes):

    run_episodes_route_similarity_raw = get_routes_similarity(size, sliding_window_sequence, routes)

    run_episodes_route_similarity_smoothed = list(moving_average(run_episodes_route_similarity_raw,SLIDING_WINDOW_SIZE_USED_FOR_SMOOTHING_C_SCORE))
    run_episodes_route_similarity_adjusted  =[np.nan, np.nan] + run_episodes_route_similarity_smoothed + [np.nan, np.nan]

    #look for the last index where the smoothed C score has dropped below the threshold
    reversed_routes = run_episodes_route_similarity_adjusted.copy()
    reversed_routes.reverse()

    try:
        temp_index = list(map(lambda i: i>200, reversed_routes)).index(True)
    except ValueError as e:
        if e.args[0] == "True is not in list": # all values are below the threshold
            temp_index=-1
        else:
            raise e

    if temp_index == 0: # no value below threshold
        index = -1
    elif temp_index == -1: # all values below threshold
        index = 0
    else:
        index = len(run_episodes_route_similarity_adjusted) - temp_index

    return index

def get_modal_target_sequence_for_run(optimal_trapline, C_score_index, routes): 

    run_episodes_targets_sequence = [] #list of an ordered list of the order targets were discovered for each sample episode in this run
    
    if C_score_index != -1:
        run_episode_routes_filtered = routes[C_score_index:]
        
        #now find model target sequence from the C_score index for each run
        for run_episode_route in run_episode_routes_filtered:
            target_sequence = get_valid_target_sequence_from_route(optimal_trapline, run_episode_route)
            run_episodes_targets_sequence.append(target_sequence)

        # get a count of al routes in the final sequence
        count = pd.Series(run_episodes_targets_sequence).value_counts().sort_values(ascending=False) 

        route = str(count.index[0])
        count = count[0]
    
    else:
        route = '[]'
        count = 0

    return route, count
    
num_runs_in_experiment = all_run_sample_episodes_in_experiment.shape[0]
num_sample_episodes_per_run = all_run_sample_episodes_in_experiment.shape[1]

# get the optimal trapline and its reverse from the MDP.  
optimal_trapline_master, optimal_trapline_reversed_master = get_optimal_trapline_for_diamond_array(MDP["targets"])

# initalise result arrays
results = pd.DataFrame()
results["route"] = np.array(num_runs_in_experiment, dtype=object) # one entry for each run
results["count"] = np.zeros(num_runs_in_experiment, dtype=int)
results["c_score_index"] = np.zeros(num_runs_in_experiment, dtype=int)
            
SLIDING_WINDOW_SIZE_USED_FOR_SMOOTHING_C_SCORE =5
SLIDING_WINDOW_SIZE_USED_FOR_COMPARING_ROUTE_SIMILARITY = 2
# get the sliding widow to use in determining if there is a stable trapline
sliding_sequence_used_for_route_similarity = get_sliding_window_sequence(SLIDING_WINDOW_SIZE_USED_FOR_COMPARING_ROUTE_SIMILARITY, num_sample_episodes_per_run)


    
for run_index in range(num_runs_in_experiment):

    run_episodes_route_observations = all_run_sample_episodes_in_experiment[run_index] # all the observations in a specific run.  i,e. all the episodes and their runs
    run_episodes_routes = get_route_index_from_observation(run_episodes_route_observations) #extract the route indexes from the route observations

    # get thw C score index for this run
    C_score_index = get_C_scores_index_for_run(MDP["size"], sliding_sequence_used_for_route_similarity, run_episodes_routes)

    # save the index value
    results.loc[run_index, 'c_score_index'] = C_score_index

    route, count = get_modal_target_sequence_for_run(optimal_trapline_master, C_score_index, run_episodes_routes)
    
    results.loc[run_index, 'route'] = route #only save the route if a stable trapline was found
    results.loc[run_index, 'count'] = count
    
    
# get a count of all the different routes of the traplines from each run
route_count_for_experiment = pd.Series(results["route"]).value_counts().sort_values(ascending=False)
#print(route_count_for_experiment)

LABEL_NO_ROUTE_FOUND = 'No route found'

# reformat data frame for plotting
df = pd.DataFrame(route_count_for_experiment)
df['count'] = df['route']
df['route'] = [loads(d) for d in df.index.to_list()]
df.index = np.arange(0, len(df))

# build x-axis labels
counter = 1
x_axis = []
for i, r in enumerate(df['route']):
    if r == []:
        x_axis.append(LABEL_NO_ROUTE_FOUND)
    else:
        x_axis.append(str(counter))
        counter += 1
df['x-axis'] = x_axis


# add nest to optimal route
optimal_trapline_inc_nest = optimal_trapline_master.copy()
optimal_trapline_inc_nest.append(MDP["nest"])

# add nest to reverse of optimal route
optimal_trapline_reversed_inc_nest = optimal_trapline_reversed_master.copy()
optimal_trapline_reversed_inc_nest.append(MDP["nest"])

# determine if each route is an optimal route
is_optimal_route = lambda x : (x == optimal_trapline_inc_nest) or (x == optimal_trapline_reversed_inc_nest)
df['optimal_route'] = [is_optimal_route(route) for route in df['route']]

#plot bar chart
fig1, ax = plt.subplots(1,1)
bar_list = ax.bar(df['x-axis'], df['count']) # plot the bar chart

use_logarithmic  = False
ax.set_xlabel('Routes')
if use_logarithmic:
    ax.set_yscale('log') # use logarithmic scale
    ax.set_ylabel('Logarithmic Count of Routes')
else:
    ax.set_ylabel('Count of Routes')

# highlight the optimal traplines if present in the results
for i in range(len(df)):
    label = df['x-axis'][i]
    route = df['route'][i]
    optimal = df['optimal_route'][i]
    if label == LABEL_NO_ROUTE_FOUND:
        bar_list[i].set_color('r')
    elif optimal:
        bar_list[i].set_color('g')

ax.set_xticklabels(df['x-axis'], rotation = 90)
fig1.tight_layout()

# drop the route count with no discernable target based route found
df = df.drop(df[df['x-axis'] == LABEL_NO_ROUTE_FOUND].index)

plot_size = int(np.ceil(np.sqrt(len(df))))
fig2, axs = plt.subplots(plot_size, plot_size, figsize=(plot_size*3, plot_size*3))

axs = np.array(axs).reshape(-1)

for i, ax in enumerate(axs):

    if i >= len(df):
        break
    # Convert string representation of route to list using json
    route = df['route'][df.index[i]]
    label= df['x-axis'][df.index[i]]
    optimal= df['optimal_route'][df.index[i]]

    plot_route(fig2, ax, MDP["size"],MDP["nest"], optimal_trapline_master, route, optimal, str(label))

plt.show()