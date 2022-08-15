
import numpy as np
from scipy.spatial import distance as distance_lib

from utils import get_sliding_window_sequence, map_index_to_coord
from trapline import cluster_common_route_segments

def get_manhattan_similarity(size, route_index1, route_index2):
    '''
    distance of zero indicates identical routes
    '''

    # work out which route is shorter
    if len(route_index1)<len(route_index2):
        shortest_route = route_index1
        longest_route = route_index2
    else:
        shortest_route = route_index2
        longest_route = route_index1

    shortest_coords = [map_index_to_coord(size,x) for x in shortest_route]
    longest_coords = [map_index_to_coord(size,x) for x in longest_route]
    
    total_distance = 0

    for i in range(len(shortest_coords)):
        
        x_p, y_p = shortest_coords[i] # x, y co-ordinate of p
        x_q, y_q = longest_coords[i] # x, y co-ordinate of q

        #calculate the l1 distance between each coordinate pair
        total_distance += np.abs(x_p - x_q) + np.abs(y_p - y_q)  
    
    # add the difference in route length
    total_distance += (len(longest_route) - len(shortest_route))
    
    return total_distance


def get_manhattan_distance(size, route):
    
    
    coords = [map_index_to_coord(size,x) for x in route]
    
    sliding_window_sequence = get_sliding_window_sequence(2, len(route))

    total_distance = 0

    for window in sliding_window_sequence:
        
        x_p, y_p = coords[window[0]] # x, y co-ordinate of p
        x_q, y_q = coords[window[1]] # x, y co-ordinate of q

        #calculate the l1 distance between each coordinate pair
        total_distance += np.abs(x_p - x_q) + np.abs(y_p - y_q)  
    
    return total_distance


def get_routes_similarity(arena_size, sliding_sequence, routes):
    '''
    determine the similarity of adjacent routes using the manhattan distance 
    (L1 norm) between each adjacent clusters of common steps in each routes
    '''

    manhattan_distances = []
    for window in sliding_sequence:
        route_index1, route_index2 = window
        route_1 = routes[route_index1]
        route_2 = routes[route_index2]

        #break routes up in clusters of commonality
        route1_segments, route2_segments = cluster_common_route_segments(route_1, route_2)

        # validate resultant clusters
        if len(route1_segments) != len(route2_segments):
            raise ValueError("Cluster segments are not the same length.  Route 1: {}.  Route 2:{}".format(route_1, route_2))

        total_segmented_manhattan_distance = 0
        for i, segment1 in enumerate(route1_segments):
            segment2 = route2_segments[i]
            
            distance = get_manhattan_similarity(arena_size, segment1, segment2)
            total_segmented_manhattan_distance += distance

        manhattan_distances.append(total_segmented_manhattan_distance)

    return manhattan_distances