import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from manhattan import get_routes_similarity
from utils import moving_average

def get_similarity_score_prime(smooth):

    smooth_d1 = np.gradient(smooth) # compute first derivative
    #smooth_d2 = np.gradient(smooth_d1) # compute second derivative
    
    return list(smooth_d1)

def get_stability_point_for_run(arena_size, sliding_window_size, sliding_window_sequence, routes, threshold=2):

    run_episodes_route_similarity_raw = get_routes_similarity(arena_size, sliding_window_sequence, routes)

    run_episodes_route_similarity_smoothed = list(moving_average(run_episodes_route_similarity_raw, sliding_window_size))

    run_episodes_route_similarity_prime =  get_similarity_score_prime(run_episodes_route_similarity_smoothed)
    
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
        index = index + 2 # to account for the 2 nans added below

    run_episodes_route_similarity_adjusted  = [np.nan, np.nan] + run_episodes_route_similarity_smoothed + [np.nan, np.nan]
    run_episodes_route_similarity_prime_adjusted = [np.nan, np.nan] +  run_episodes_route_similarity_prime + [np.nan, np.nan]

    return index, run_episodes_route_similarity_adjusted, run_episodes_route_similarity_prime_adjusted
    