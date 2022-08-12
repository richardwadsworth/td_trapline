from utils import map_index_to_coord
import numpy as np

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