#plot learning over n experiments, showing error bars

    #find C for all sample routes
    #smooth C using sliding window
    #plot
    #traverse graph until C < threshold
    #find modal target sequence from that point

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from mlflow_utils import get_experiment_runs_data
from utils import get_sliding_window_sequence
from trapline import get_optimal_trapline_for_diamond_array, get_routes_similarity, get_valid_target_sequence_from_route, RouteType
from plots import plot_route, plot_trapline_distribution
 

experiment_name = "analyse_32bed68ecebc40849485df2ad8d5958f_10_medium_positive_array_chittka" #best 10 positive chittka, 200 episodes
#experiment_name = "analyse_dbe7b192cd70476dbd59e2e65153c1a5_10_medium_negative_array_chittka" #best 10 negative chittka, 200 episodes


data, plot_rate = get_experiment_runs_data(experiment_name) 
all_run_sample_episodes_in_experiment = data["observations"]
all_run_sample_done_in_experiment = data["done"]
MDP = data["MDP"]


# pickle.dump( all_run_sample_episodes_in_experiment, open( "all_run_sample_episodes_in_experiment.p", "wb" ) )
# pickle.dump( all_run_sample_done_in_experiment, open( "all_run_sample_done_in_experiment.p", "wb" ) )
# pickle.dump( MDP, open( "all_MDP.p", "wb" ) )

# all_run_sample_episodes_in_experiment = pickle.load( open( "all_run_sample_episodes_in_experiment.p", "rb" ) )
# all_run_sample_done_in_experiment = pickle.load( open( "all_run_sample_done_in_experiment.p", "rb" ) )
# MDP = pickle.load( open( "all_MDP.p", "rb" ) )


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

    return index, run_episodes_route_similarity_adjusted

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
results["route"] = np.empty((num_runs_in_experiment,), dtype=object) # one entry for each run
results["count"] = np.zeros(num_runs_in_experiment, dtype=int)
results["c_score_index"] = np.zeros(num_runs_in_experiment, dtype=int)
            
SLIDING_WINDOW_SIZE_USED_FOR_SMOOTHING_C_SCORE =5
SLIDING_WINDOW_SIZE_USED_FOR_COMPARING_ROUTE_SIMILARITY = 2
# get the sliding widow to use in determining if there is a stable trapline
sliding_sequence_used_for_route_similarity = get_sliding_window_sequence(SLIDING_WINDOW_SIZE_USED_FOR_COMPARING_ROUTE_SIMILARITY, num_sample_episodes_per_run)

route_c_scores = []
for run_index in range(num_runs_in_experiment):

    run_episodes_route_observations = all_run_sample_episodes_in_experiment[run_index] # all the observations in a specific run.  i,e. all the episodes and their runs
    run_episodes_routes = get_route_index_from_observation(run_episodes_route_observations) #extract the route indexes from the route observations

    # get thw C score index for this run
    C_score_index, run_episodes_route_similarity_adjusted = get_C_scores_index_for_run(MDP["size"], sliding_sequence_used_for_route_similarity, run_episodes_routes)
    
    # save the index value and smoothed  scores
    route_c_scores.append(run_episodes_route_similarity_adjusted)
    results.loc[run_index, 'c_score_index'] = C_score_index

    route, count = get_modal_target_sequence_for_run(optimal_trapline_master, C_score_index, run_episodes_routes)
    
    results.loc[run_index, 'route'] = route #only save the route if a stable trapline was found
    results.loc[run_index, 'count'] = count
    
    
# get a count of all the different routes of the traplines from each run
route_count_for_experiment = pd.Series(results["route"]).value_counts().sort_values(ascending=False)
#print(route_count_for_experiment)

import json
import seaborn as sns

def plot_c_Scores(experiment_name, smoothed):
    fig, ax = plt.subplots(1,1)
    sns.set_theme(style="whitegrid")

    for route in smoothed:
        xs = np.arange(0, len(route)) * 5
       
        alpha = 0.7
        ax.plot(xs, route, alpha=alpha)

    
    # calculate the mean C score across all runs
    df = pd.DataFrame(smoothed)
    mean = df.mean() 
    xs = np.arange(0, len(mean)) * 5
    alpha = 1
    ax.plot(xs, mean, alpha=alpha, lw=2, color='black', label="Mean C score")
    ax.legend(loc='upper right')

    ax.set_xlabel('Episode')
    ax.set_ylabel('(Smoothed) C Score')
    fig.suptitle("Route C Score by Episode")
    ax.set_title(experiment_name, fontsize=10)

    plt.pause(0.00000000001)

plot_c_Scores(experiment_name, route_c_scores)

plot_trapline_distribution(experiment_name, MDP, route_count_for_experiment, optimal_trapline_master, optimal_trapline_reversed_master)