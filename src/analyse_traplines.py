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
from json import loads

from  manhattan import get_manhattan_distance
from mlflow_utils import get_experiment_runs_data
from utils import get_sliding_window_sequence
from trapline import get_optimal_trapline_for_diamond_array, get_valid_target_sequence_from_route, RouteType
from plots import plot_trapline_distribution, plot_c_Scores, plot_c_score_stability_distribution, plot_target_sequence_length_distribution
from c_score import get_C_scores_index_for_run
 

#experiment_name = "analyse_d7f2c7c777164160bd5ac6bbfefb0a71_10_medium_negative_array_ohashi_gs_1000_runs" #best 10 negative ohashi, 200 episodes, 1000 runs
#experiment_name = "analyse_3cb6d9c1e8c646188668a059a9190d6c_10_medium_positive_array_ohashi_gs_1000_runs" #best 10 positive ohashi, 200 episodes, 1000 runs
#experiment_name = "analyse_bc6e223900244462b7898b0b511a9a4b_mrp_10_negative_array_ohashi_gs_1000_runs" #best 10 negative ohashi, 200 episodes, 1000 runs
experiment_name = "analyse_cb41737061ea435e8b43abfddc0258cf_mrp_10_positive_array_ohashi_gs_1000_runs" #best 10 positive ohashi, 200 episodes, 1000 runs

artifact_path = "artifacts"

data = get_experiment_runs_data(experiment_name) 

#use pickle to cache data to speed up r&D
#pickle.dump( data, open( experiment_name + "_data.p", "wb" ) )
#data = pickle.load( open( experiment_name + "_data.p", "rb" ) )

all_run_sample_episodes_in_experiment = data["observations"]
all_run_sample_done_in_experiment = data["done"]
MRP = data["MRP"]
sample_rate = int(data["params"][0]["plot_rate"])

def get_route_index_from_observation(route_observations):
    route_indexes = [[observation[0] for observation in observations] for observations in route_observations]
    return route_indexes


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

        target_sequence = str(count.index[0])
        target_sequence_count = count[0]
    
    else:
        target_sequence = '[]'
        target_sequence_count = 0

    return target_sequence, target_sequence_count
    
num_runs_in_experiment = all_run_sample_episodes_in_experiment.shape[0]
num_sample_episodes_per_run = all_run_sample_episodes_in_experiment.shape[1]

# get the optimal trapline and its reverse from the MRP.  
optimal_trapline_master, optimal_trapline_reversed_master = get_optimal_trapline_for_diamond_array(MRP["targets"])

SLIDING_WINDOW_SIZE_USED_FOR_SMOOTHING_C_SCORE =5
SLIDING_WINDOW_SIZE_USED_FOR_COMPARING_ROUTE_SIMILARITY = 2
C_SCORE_STABILITY_THRESHOLD = 0

# initalise result arrays
results = pd.DataFrame()
results["target_sequence"] = np.empty((num_runs_in_experiment,), dtype=object) # one entry for each run
results["target_sequence_count"] = np.zeros(num_runs_in_experiment, dtype=int)
results["c_score_stability_index"] = np.zeros(num_runs_in_experiment, dtype=int)
            
# get the sliding widow to use in determining if there is a stable trapline
sliding_sequence_used_for_route_similarity = get_sliding_window_sequence(SLIDING_WINDOW_SIZE_USED_FOR_COMPARING_ROUTE_SIMILARITY, num_sample_episodes_per_run)

route_c_scores = []
for run_index in range(num_runs_in_experiment):

    run_episodes_route_observations = all_run_sample_episodes_in_experiment[run_index] # all the observations in a specific run.  i,e. all the episodes and their runs
    run_episodes_routes = get_route_index_from_observation(run_episodes_route_observations) #extract the route indexes from the route observations

    # get thw C score index for this run
    C_score_index, run_episodes_route_similarity_adjusted, run_episodes_route_similarity_prime_adjusted = get_C_scores_index_for_run(int(MRP["size"]), SLIDING_WINDOW_SIZE_USED_FOR_SMOOTHING_C_SCORE, sliding_sequence_used_for_route_similarity, run_episodes_routes, C_SCORE_STABILITY_THRESHOLD)
    
    # save the index value and smoothed  scores
    route_c_scores.append((run_episodes_route_similarity_adjusted, run_episodes_route_similarity_prime_adjusted))
    results.loc[run_index, 'c_score_stability_index'] = C_score_index

    target_sequence, target_sequence_count = get_modal_target_sequence_for_run(optimal_trapline_master, C_score_index, run_episodes_routes)
    
    results.loc[run_index, "target_sequence"] = target_sequence #only save the target_sequence if a stable trapline was found
    results.loc[run_index, "target_sequence_count"] = target_sequence_count
    
results["c_score_indexes"] = [x[0] for x in route_c_scores]
results["c_score_indexes_rate_of_change"] = [x[1] for x in route_c_scores]

#save results to file
results.to_csv("artifacts/" + experiment_name + "_results")

# get a count of all the different routes of the traplines from each run
route_count_for_experiment = pd.Series(results["target_sequence"]).value_counts().sort_values(ascending=False)
#print(route_count_for_experiment)

# reformat data frame for plotting
df_target_sequence_data_for_experiment = pd.DataFrame(route_count_for_experiment)
df_target_sequence_data_for_experiment["target_sequence_count"] = df_target_sequence_data_for_experiment["target_sequence"]
df_target_sequence_data_for_experiment["target_sequence"] = [loads(d) for d in df_target_sequence_data_for_experiment.index.to_list()]
df_target_sequence_data_for_experiment.index = np.arange(0, len(df_target_sequence_data_for_experiment))


distances = []
for i in range (len(df_target_sequence_data_for_experiment)):
    row = df_target_sequence_data_for_experiment.iloc[i]
    #calculate the manhattan length of this target sequence
    distance = get_manhattan_distance(int(MRP["size"]), row["target_sequence"])
    distances.append(distance)

df_target_sequence_data_for_experiment['sequence_manhattan_length'] = distances
#normalise the sequence_manhattan_length using the optimal sequence length to allow for comparison with other target arrays
optimal_target_sequence_length = get_manhattan_distance(int(MRP["size"]), optimal_trapline_master) # we can use either optimal route, clockwise or anti clockwise to determine the optimal length
df_target_sequence_data_for_experiment['sequence_manhattan_length'] = df_target_sequence_data_for_experiment['sequence_manhattan_length']/optimal_target_sequence_length

plot_target_sequence_length_distribution(experiment_name, artifact_path, num_runs_in_experiment, df_target_sequence_data_for_experiment)

plot_c_Scores(experiment_name, artifact_path, sample_rate, results["c_score_indexes"], results["c_score_indexes_rate_of_change"])

plot_c_score_stability_distribution(experiment_name, artifact_path, sample_rate, C_SCORE_STABILITY_THRESHOLD, list(results['c_score_stability_index']))

plot_trapline_distribution(experiment_name, artifact_path, num_runs_in_experiment, MRP, df_target_sequence_data_for_experiment, optimal_trapline_master, optimal_trapline_reversed_master)

plt.show()