import numpy as np
import pandas as pd

from utils import sliding_window
from manhattan import get_manhattan_distance

def get_optimal_trapline_for_diamond_array(targets):
    '''
    calculate the resevered MDP.  
    Note:  Only works for diamonds where the first target is a corner
    '''
    #identify the optimal trapline
    optimal_trapline = [observation[0] for observation in targets]

    # calculate the resevered MDP.
    remaining = optimal_trapline[1:]
    remaining.reverse()
    optimal_trapline_reversed = [optimal_trapline[0]] + remaining #get the optimal trapline the other direction

    return optimal_trapline, optimal_trapline_reversed


def get_trapline_for_run_using_route_distribution(sliding_sequence, routes):
        '''
        determine if a stable trapline has developed
        '''
        sample_dist = np.ndarray(((len(sliding_sequence))), dtype = object) # array used to record distributions

        for i, window in enumerate(sliding_sequence):
            distribution = {} # use a dict to aid grouping the counts
            for episode_sample in window:

                key = str(routes[episode_sample])

                # count the occurrences of each route in each siding window
                distribution[key] = distribution.get(key, 0) + 1
 
            X =  np.array([[key, value] for key, value in distribution.items()], dtype=object)
            X[:, 1] = X[:, 1]/np.sum(X[:, 1]) #calculate the distribution of the counts (in scale of 0 to 1)
            sample_dist[i] = X # save the routes and their distribution for this sliding window

        distributions = [sample[:,1] for sample in sample_dist]
        max_indexes = [np.argmax(dist) for dist in distributions]
        
        sequence = []
        for i, index in enumerate(max_indexes):
            distribution = sample_dist[i]
            route = distribution[index][0]
            sequence.append(route)

        
        # get a count of al routes in the final sequence
        count = pd.Series(sequence).value_counts().sort_values(ascending=False) 

        route = str(count.index[0])
        count = count[0]

        return route, count


def is_stable_trapline(arena_size, sliding_sequence, routes, stability_threshold=100):
    '''
    determine if a stable trapline has developed using the manhattan distance 
    (L1 norm) between each adjacent routes for the last X episode samples in a run
    '''
    
    # compare the manhattan distance of all adjacent routes
    manhattan_distances = []
    for window in sliding_sequence:
        route_index1, route_index2 = window
        route_1 = routes[route_index1]
        route_2 = routes[route_index2]
        distance = get_manhattan_distance(arena_size, route_1, route_2)
        manhattan_distances.append(distance)

    # compare the mean sum of manhattan distances with the stability threshold
    mean_distance = np.mean(manhattan_distances)
    print("Mean Manhattan distance: {}".format(mean_distance))

    if mean_distance <= stability_threshold:
        trapline_found = True
    else:
        trapline_found = False

    return trapline_found


def get_ordered_target_list_for_episode(optimal_trapline, nest_index, route):

    MIN_NUM_TARGETS_FOUND_TO_CONSIDER_AS_TRAPLINE = 2 # the minimum number of targets that must be found in an episode to be considered a trapline

    trapline_lookup = optimal_trapline.copy()

    episode_target_list = [] # an ordered list of the order targets were discovered
    
    for arena_grid_index in route:
        if arena_grid_index in trapline_lookup: # is this arena_grid_index a target?
            episode_target_list.append(arena_grid_index) # record the order the target was found
            trapline_lookup.remove(arena_grid_index) #remove the target from the look up in case it was visited again

    #if this episode found the minimum number of targets AND then found the nest, then this episode should be considered
    # for further trapline analysis
    if len(episode_target_list) > MIN_NUM_TARGETS_FOUND_TO_CONSIDER_AS_TRAPLINE and route[-1] ==nest_index:
        # a valid route was found in this episode

        episode_target_list.append(nest_index) # add the nest as the last discovered "target"

        # record the ordered list of targets discovered for this episode
        return episode_target_list
    else:
        # a valid route was NOT found in this episode
        return []