'''
 Python script to analyse the results of a learning simulation.  Specifically
  to analyse the 1000 simulations generated by runner_analyse.py


 Call this script passing the in the experiment name as the argument.

 e.g. 
 python .src/analyse_traplines.py analyse_a628d5c7a59047629b7721ac09455aea_mrp_10_positive_array_ohashi_gs_1000_runs
 
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
from json import loads

from  manhattan import get_manhattan_distance
from mlflow_utils import get_experiment_runs_data
from utils import get_sliding_window_sequence
from trapline import get_optimal_trapline_for_diamond_array, get_valid_target_sequence_from_route, TargetSequenceType
from plots import plot_trapline_distribution, plot_similarity_index, plot_trapline_stability_distribution, plot_target_sequence_length_distribution, plot_similarity_index_distribution
from similarity_score import get_stability_point_for_run
 

def main():

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    args = parser.parse_args()
    experiment_name = args.experiment_name

    artifact_path = "artifacts"
    
    data = get_experiment_runs_data(experiment_name) 

    # #use pickle to cache data to speed up r&D
    #pickle.dump( data, open( experiment_name + "_data.p", "wb" ) )
    #data = pickle.load( open( experiment_name + "_data.p", "rb" ) )

    all_run_sample_episodes_in_experiment = data["observations"]
    MRP = data["MRP"]
    sample_rate = int(data["params"][0]["plot_rate"])

    def get_route_index_from_observation(route_observations):
        route_indexes = [[observation[0] for observation in observations] for observations in route_observations]
        return route_indexes


    def get_modal_target_sequence_for_run(optimal_trapline, stability_point, routes): 
        """
            get the modal target sequence from a run of episodes

        Args:
            optimal_trapline: a sequence of targets in an optimal order
            stability_point: the stability point of the run
            routes: the route in the run

        Returns:
            a sequence of targets tha the agent discovered
        """

        run_episodes_targets_sequence = [] #list of an ordered list of the order targets were discovered for each sample episode in this run
        
        if stability_point != -1:
            run_episode_routes_filtered = routes[stability_point:] # get all routes after the point of stability
            
            #now find model target sequence from the similarity_score index for each run
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
    optimal_trapline_master = get_optimal_trapline_for_diamond_array(MRP["targets"])

    SLIDING_WINDOW_SIZE_USED_FOR_SMOOTHING_SIMILARITY_SCORE =5
    SLIDING_WINDOW_SIZE_USED_FOR_COMPARING_ROUTE_SIMILARITY = 2
    RATE_OF_CHANGE_OF_STABILITY_THRESHOLD = 0

    # initalise result arrays
    results = pd.DataFrame()
    results["target_sequence"] = np.empty((num_runs_in_experiment,), dtype=object) # one entry for each run
    results["target_sequence_count"] = np.zeros(num_runs_in_experiment, dtype=int)
    results["stability_point"] = np.zeros(num_runs_in_experiment, dtype=int)
                
    # get the sliding widow to use in determining if there is a stable trapline
    sliding_sequence_used_for_route_similarity = get_sliding_window_sequence(SLIDING_WINDOW_SIZE_USED_FOR_COMPARING_ROUTE_SIMILARITY, num_sample_episodes_per_run)

    print("Analysing data.  This can take a few minutes ...")

    route_similarity_scores = []
    for run_index in range(num_runs_in_experiment):

        run_episodes_route_observations = all_run_sample_episodes_in_experiment[run_index] # all the observations in a specific run.  i,e. all the episodes and their runs
        run_episodes_routes = get_route_index_from_observation(run_episodes_route_observations) #extract the route indexes from the route observations

        # get thw C score index for this run
        stability_point, run_episodes_route_similarity_adjusted, run_episodes_route_similarity_prime_adjusted = get_stability_point_for_run(int(MRP["size"]), SLIDING_WINDOW_SIZE_USED_FOR_SMOOTHING_SIMILARITY_SCORE, sliding_sequence_used_for_route_similarity, run_episodes_routes, RATE_OF_CHANGE_OF_STABILITY_THRESHOLD)
        
        # save the index value and smoothed  scores
        route_similarity_scores.append((run_episodes_route_similarity_adjusted, run_episodes_route_similarity_prime_adjusted))
        results.loc[run_index, 'stability_point'] = stability_point

        target_sequence, target_sequence_count = get_modal_target_sequence_for_run(optimal_trapline_master, stability_point, run_episodes_routes)
        
        results.loc[run_index, "target_sequence"] = target_sequence #only save the target_sequence if a stable trapline was found
        results.loc[run_index, "target_sequence_count"] = target_sequence_count
        
    results["route_similarity_score"] = [x[0] for x in route_similarity_scores] # hack.  use list comprehension as I can't work out how to set the df row to a list
    results["route_similarity_score_rate_of_change"] = [x[1] for x in route_similarity_scores] # hack.  use list comprehension as I can't work out how to set the df row to a list

    #save results to file
    results.to_csv("artifacts/" + experiment_name + "_results.csv")

    # get a count of all the different routes of the traplines from each run
    route_count_for_experiment = pd.Series(results["target_sequence"]).value_counts().sort_values(ascending=False)
    #print(route_count_for_experiment)

    # reformat data frame for plotting
    df_target_sequence_data_for_experiment = pd.DataFrame(route_count_for_experiment)
    df_target_sequence_data_for_experiment["target_sequence_count"] = df_target_sequence_data_for_experiment["target_sequence"]
    df_target_sequence_data_for_experiment["target_sequence"] = [loads(d) for d in df_target_sequence_data_for_experiment.index.to_list()]
    df_target_sequence_data_for_experiment.index = np.arange(0, len(df_target_sequence_data_for_experiment))


    def wrap_sequence_with_nest(nest_index, sequence):
        return [nest_index] + sequence + [nest_index]

    distances = []
    target_sequences_including_nest = []
    for i in range (len(df_target_sequence_data_for_experiment)):
        row = df_target_sequence_data_for_experiment.iloc[i]
        #calculate the manhattan length of this target sequence
        if row["target_sequence"] != []:
            target_sequence_including_nest = wrap_sequence_with_nest(MRP["nest"],row["target_sequence"])
        else:
            target_sequence_including_nest = []
        distance = get_manhattan_distance(int(MRP["size"]), target_sequence_including_nest)
        target_sequences_including_nest.append(target_sequence_including_nest)
        distances.append(distance)

    df_target_sequence_data_for_experiment['target_sequence_including_nest'] = target_sequences_including_nest
    df_target_sequence_data_for_experiment['sequence_manhattan_length'] = distances
    #normalise the sequence_manhattan_length using the optimal sequence length to allow for comparison with other target arrays
    optimal_trapline_master_including_nest = wrap_sequence_with_nest(MRP["nest"],optimal_trapline_master)
    optimal_target_including_nest_sequence_length = get_manhattan_distance(int(MRP["size"]), optimal_trapline_master_including_nest) # we can use either optimal route, clockwise or anti clockwise to determine the optimal length
    df_target_sequence_data_for_experiment['sequence_manhattan_length'] = df_target_sequence_data_for_experiment['sequence_manhattan_length']/optimal_target_including_nest_sequence_length

    plot_target_sequence_length_distribution(experiment_name, artifact_path, num_runs_in_experiment, df_target_sequence_data_for_experiment)
    
    plot_similarity_index_distribution(experiment_name, artifact_path, sample_rate, num_sample_episodes_per_run, results["route_similarity_score"],50) # last 50 episodes
    
    plot_similarity_index(experiment_name, artifact_path, sample_rate, results["route_similarity_score"], results["route_similarity_score_rate_of_change"])

    plot_trapline_stability_distribution(experiment_name, artifact_path, sample_rate, RATE_OF_CHANGE_OF_STABILITY_THRESHOLD, list(results['stability_point']))

    plot_trapline_distribution(experiment_name, artifact_path, num_runs_in_experiment, MRP, df_target_sequence_data_for_experiment, optimal_trapline_master)

    plt.show()

if __name__ == "__main__":
   main()