from enum import Enum

class RouteType(Enum):
    Incomplete = 0 # not all targets found
    SubOptimal = 1 # all targets found but not optimal, TSP route
    Optimal = 2 # all targets found using optimal route



def get_optimal_trapline_for_diamond_array(targets):
    '''
    calculate the resevered MRP.  
    Note:  Only works for diamonds where the first target is a corner
    '''
    #identify the optimal trapline
    optimal_trapline = [observation[0] for observation in targets]

    # calculate the resevered MRP.
    remaining = optimal_trapline[1:]
    remaining.reverse()
    optimal_trapline_reversed = [optimal_trapline[0]] + remaining #get the optimal trapline the other direction

    return optimal_trapline, optimal_trapline_reversed

def get_valid_target_sequence_from_route(optimal_trapline, route):

    MIN_NUM_TARGETS_FOUND_TO_CONSIDER_AS_TRAPLINE = len(optimal_trapline) # the minimum number of targets that must be found in an episode to be considered a trapline

    trapline_lookup = optimal_trapline.copy()

    episode_target_list = [] # an ordered lis

    for arena_grid_index in route:
        if arena_grid_index in trapline_lookup: # is this arena_grid_index a target?
            episode_target_list.append(arena_grid_index) # record the order the target was found
            trapline_lookup.remove(arena_grid_index) #remove the target from the look up in case it was visited again

    #if this episode found the minimum number of targets AND then found the nest, then this episode should be considered
    # for further trapline analysis
    if len(episode_target_list) >= MIN_NUM_TARGETS_FOUND_TO_CONSIDER_AS_TRAPLINE:
        # a valid route was found in this episode

        # record the ordered list of targets discovered for this episode
        return episode_target_list
    else:
        # a valid route was NOT found in this episode
        return []
        

def cluster_common_route_segments(route_1, route_2):
    '''
    take two routes and split the routes in to clusters of common and uncommon routes
    '''
    pointer_1=0
    pointer_2=0

    route_1_segments = []
    route_2_segments = []

    last_common_1 = 0
    last_common_2 = 0

    while pointer_1<len(route_1):

        if route_1[pointer_1]==route_2[pointer_2]:
            # the route position is the same.  

            if (pointer_1+1 == len(route_1) or pointer_2+1 == len(route_2)) or \
                ((route_1[pointer_1-1]==route_2[pointer_2-1]) and (route_1[pointer_1+1]!=route_2[pointer_2+1])) or \
                ((route_1[pointer_1-1]!=route_2[pointer_2-1]) and (route_1[pointer_1]!=route_2[pointer_2-1]))   :
                #the next indexes do not match
            
                if (pointer_1+1 != len(route_1)) and (pointer_2+1 == len(route_2)):
                    # Found matching indexes but we're at the end of route 2 so no where
                    # else to go after this so need to include the remainder of route 1 in 
                    # segment by setting the route_1 pointer to the end of the route.
                    pointer_1 = len(route_1)-1

                #create segments for each route
                route_1_segment = route_1[last_common_1: pointer_1+1]
                route_2_segment = route_2[last_common_2: pointer_2+1]

                route_1_segments.append(route_1_segment)
                route_2_segments.append(route_2_segment)
            
                last_common_1 = pointer_1
                last_common_2 = pointer_2
            
            pointer_1+=1
            pointer_2+=1
            
        elif (pointer_2 == len(route_2)-1) and (pointer_1 != len(route_1)-1): 
            # we are at the end of route2 and no additional common index found.
            
            # move on to next index in route1 to continue the search
            pointer_1+=1
            pointer_2 = last_common_2+1

        elif (pointer_1== len(route_1)-1) and (pointer_2 == len(route_2)-1):

            # we're at the end of both routes. create final segments for each route
            route_1_segment = route_1[last_common_1: pointer_1+1]
            route_2_segment = route_2[last_common_2: pointer_2+1]

            route_1_segments.append(route_1_segment)
            route_2_segments.append(route_2_segment)

            break 

        else:
            pointer_2+=1
            
    return route_1_segments, route_2_segments




