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
import os

from mlflow_utils import get_experiment_runs_data
from utils import get_sliding_window_sequence
from trapline import get_optimal_trapline_for_diamond_array, get_routes_similarity, get_valid_target_sequence_from_route, RouteType
from plots import plot_route, plot_trapline_distribution, plot_c_Scores, plot_c_score_stability_distribution
from c_score import get_c_score_prime
 
artifact_path = "sussex/Dissertation/artifacts"

#experiment_name = "analyse_32bed68ecebc40849485df2ad8d5958f_10_medium_positive_array_chittka_100_runs" #best 10 positive chittka, 200 episodes, 100 runs
experiment_name = "analyse_dbe7b192cd70476dbd59e2e65153c1a5_10_medium_negative_array_chittka_100_runs" #best 10 negative chittka, 200 episodes, 100 runs

#experiment_name = "analyse_32bed68ecebc40849485df2ad8d5958f_10_medium_positive_array_chittka_1000_runs" #best 10 positive chittka, 200 episodes, 1000 runs
#experiment_name = "analyse_dbe7b192cd70476dbd59e2e65153c1a5_10_medium_negative_array_chittka_1000_runs" #best 10 negative chittka, 200 episodes, 1000 runs

data, sample_rate = get_experiment_runs_data(experiment_name) 
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

def get_C_scores_index_for_run(size, sliding_window_sequence, routes, threshold=2):

    run_episodes_route_similarity_raw = get_routes_similarity(size, sliding_window_sequence, routes)

    run_episodes_route_similarity_smoothed = list(moving_average(run_episodes_route_similarity_raw,SLIDING_WINDOW_SIZE_USED_FOR_SMOOTHING_C_SCORE))

    run_episodes_route_similarity_prime =  get_c_score_prime(run_episodes_route_similarity_smoothed)
    
    # run_episodes_route_similarity_smoothed

    
    #look for the last index where the smoothed C score has dropped below the threshold
    reversed_graph = run_episodes_route_similarity_prime.copy()
    reversed_graph.reverse()

    try:
        # rate of change in similarity threshold 
        temp_index = list(map(lambda i: i>threshold or i<-threshold, reversed_graph)).index(True)
    except ValueError as e:
        if e.args[0] == "True is not in list": # no values are within the threshold
            temp_index=-1
        else:
            raise e

    if temp_index == 0: # the last value (first value in the reverse list) in the list is outside the threshold
        index = -1
    elif temp_index == -1: # all values are within the threshold
        index = 0
    else:
        index = len(run_episodes_route_similarity_prime) - temp_index
        index = index - 2 # to account for the 2 nans added below

    run_episodes_route_similarity_adjusted  = [np.nan, np.nan] + run_episodes_route_similarity_smoothed + [np.nan, np.nan]
    run_episodes_route_similarity_prime_adjusted = [np.nan, np.nan] +  run_episodes_route_similarity_prime + [np.nan, np.nan]

    return index, run_episodes_route_similarity_adjusted, run_episodes_route_similarity_prime_adjusted


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

SLIDING_WINDOW_SIZE_USED_FOR_SMOOTHING_C_SCORE =5
SLIDING_WINDOW_SIZE_USED_FOR_COMPARING_ROUTE_SIMILARITY = 2
C_SCORE_STABILITY_THRESHOLD = 0

# initalise result arrays
results = pd.DataFrame()
results["route"] = np.empty((num_runs_in_experiment,), dtype=object) # one entry for each run
results["count"] = np.zeros(num_runs_in_experiment, dtype=int)
results["c_score_stability_index"] = np.zeros(num_runs_in_experiment, dtype=int)
            
# get the sliding widow to use in determining if there is a stable trapline
sliding_sequence_used_for_route_similarity = get_sliding_window_sequence(SLIDING_WINDOW_SIZE_USED_FOR_COMPARING_ROUTE_SIMILARITY, num_sample_episodes_per_run)

route_c_scores = []
for run_index in range(num_runs_in_experiment):

    run_episodes_route_observations = all_run_sample_episodes_in_experiment[run_index] # all the observations in a specific run.  i,e. all the episodes and their runs
    run_episodes_routes = get_route_index_from_observation(run_episodes_route_observations) #extract the route indexes from the route observations

    # get thw C score index for this run
    C_score_index, run_episodes_route_similarity_adjusted, run_episodes_route_similarity_prime_adjusted = get_C_scores_index_for_run(MDP["size"], sliding_sequence_used_for_route_similarity, run_episodes_routes, C_SCORE_STABILITY_THRESHOLD)
    
    # save the index value and smoothed  scores
    route_c_scores.append((run_episodes_route_similarity_adjusted, run_episodes_route_similarity_prime_adjusted))
    results.loc[run_index, 'c_score_stability_index'] = C_score_index

    route, count = get_modal_target_sequence_for_run(optimal_trapline_master, C_score_index, run_episodes_routes)
    
    results.loc[run_index, 'route'] = route #only save the route if a stable trapline was found
    results.loc[run_index, 'count'] = count
    
results["c_score_indexes"] = [x[0] for x in route_c_scores]
results["c_score_indexes_rate_of_change"] = [x[1] for x in route_c_scores]

results.to_csv("sussex/Dissertation/artifacts/" + experiment_name + "_results")

# get a count of all the different routes of the traplines from each run
route_count_for_experiment = pd.Series(results["route"]).value_counts().sort_values(ascending=False)
#print(route_count_for_experiment)

# reformat data frame for plotting
from json import loads
df_route_count_for_experiment = pd.DataFrame(route_count_for_experiment)
df_route_count_for_experiment['count'] = df_route_count_for_experiment['route']
df_route_count_for_experiment['route'] = [loads(d) for d in df_route_count_for_experiment.index.to_list()]
df_route_count_for_experiment.index = np.arange(0, len(df_route_count_for_experiment))

from  manhattan import get_manhattan_distance, get_euclidean_distance
distances = []
for i in range (len(df_route_count_for_experiment)):
    row = df_route_count_for_experiment.iloc[i]
    #calculate the manhattan length of this target sequence
    distance = get_manhattan_distance(17, row['route'])
    distances.append(distance)

df_route_count_for_experiment['sequence_manhattan_length'] = distances

distances = []
for i in range (len(df_route_count_for_experiment)):
    row = df_route_count_for_experiment.iloc[i]
    #calculate the manhattan length of this target sequence
    distance = get_euclidean_distance(17, row['route'])
    distances.append(distance)

df_route_count_for_experiment['sequence_euclidean_length'] = distances


from c_score import get_c_score_prime

plot_c_Scores(experiment_name, sample_rate, results["c_score_indexes"], results["c_score_indexes_rate_of_change"])

plot_c_score_stability_distribution(experiment_name, sample_rate, C_SCORE_STABILITY_THRESHOLD, list(results['c_score_stability_index']))

plot_trapline_distribution(experiment_name, num_runs_in_experiment, MDP, df_route_count_for_experiment, optimal_trapline_master, optimal_trapline_reversed_master)

plt.show()