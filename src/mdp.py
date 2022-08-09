import numpy as np
from json import dumps
from utils import map_coord_to_index

dump_mdp  = lambda x : dumps(x)

##########################################################################
## NOTE the order of the MDP.targets list is assumed to be the optimal 
#       trapline (which can also be reversed).
##########################################################################

def get_1_small_test_array():
    # small neutral array, offest nest
    name = "6_small_neutral_array_offset"
    size = 4
    MDP = {"nest":map_coord_to_index(size, 1, 1),
            "targets": [(map_coord_to_index(size, 2, 2),1.0)
                    ]
        }
    return size, dump_mdp(MDP), name

def get_6_small_neutral_array():
    # small neutral array, offest nest
    name = "6_small_neutral_array_offset"
    size = 8
    MDP = {"nest":map_coord_to_index(size, 1, 1),
            "targets": [(map_coord_to_index(size, 2, 2),1.0), 
                        (map_coord_to_index(size, 4, 2),1.0),
                        (map_coord_to_index(size, 6, 4),1.0),
                        (map_coord_to_index(size, 6, 6),1.0),
                        (map_coord_to_index(size, 4, 6),1.0),
                        (map_coord_to_index(size, 2, 4),1.0)
                    ]
        }
    return size, dump_mdp(MDP), name

def get_6_small_positive_array():
    # small positive array, offest nest
    name = "6_small_positive_array_offset"
    size = 8
    MDP = {"nest":map_coord_to_index(size, 1, 1),
            "targets": [
                    (map_coord_to_index(size, 2, 2),1.0), 
                    (map_coord_to_index(size, 4, 2),1.0),
                    (map_coord_to_index(size, 5, 3),1.0),
                    (map_coord_to_index(size, 5, 5),1.0),
                    (map_coord_to_index(size, 3, 5),1.0),
                    (map_coord_to_index(size, 2, 4),1.0)
                    ]
        }
    return size, dump_mdp(MDP), name


def get_6_medium_positive_array():
    # medium positive array, offest nest
    name = "6_medium_positive_array_offset"
    size = 12
    MDP = {"size": size,
            "nest":map_coord_to_index(size, 2, 2),
            "targets": [
                    (map_coord_to_index(size, 4, 4),1.0), 
                    (map_coord_to_index(size, 7, 4),1.0),
                    (map_coord_to_index(size, 9, 6),1.0),
                    (map_coord_to_index(size, 9, 9),1.0),
                    (map_coord_to_index(size, 6, 9),1.0),
                    (map_coord_to_index(size, 4, 7),1.0)
                    ]
        }
    return size, dump_mdp(MDP), name

def get_6_large_positive_array():
    # large positive array, offest nest
    name = "6_large_positive_array_offset"
    size = 19
    MDP = {"size": size,
            "nest":map_coord_to_index(size, 3, 3),
            "targets": [
                    (map_coord_to_index(size, 6, 6),1.0), 
                    (map_coord_to_index(size, 11, 6),1.0),
                    (map_coord_to_index(size, 14, 9),1.0),
                    (map_coord_to_index(size, 9, 14),1.0),
                    (map_coord_to_index(size, 14, 14),1.0),
                    (map_coord_to_index(size, 6, 11),1.0)
                    ]
        }
    return size, dump_mdp(MDP), name



def get_10_medium_positive_array():
    # medium positive array, offest nest
    name = "10_medium_positive_array_offset"
    size = 16
    MDP = {"size": size,
            "nest":map_coord_to_index(size, 2, 2),
            "targets": [
                    (map_coord_to_index(size, 4, 4),1.0), 
                    (map_coord_to_index(size, 7, 4),1.0),
                    (map_coord_to_index(size, 9, 6),1.0),
                    (map_coord_to_index(size, 11, 8),1.0),
                    (map_coord_to_index(size, 13, 10),1.0),
                    (map_coord_to_index(size, 13, 13),1.0),
                    (map_coord_to_index(size, 10, 13),1.0),
                    (map_coord_to_index(size, 8, 11),1.0),
                    (map_coord_to_index(size, 6, 9),1.0),
                    (map_coord_to_index(size, 4, 7),1.0)
                    ]
        }
    return size, dump_mdp(MDP), name