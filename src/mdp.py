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
    name = "1_small_test_array"
    size = 4
    MDP = {"nest":map_coord_to_index(size, 1, 1),
            "targets": [(map_coord_to_index(size, 2, 2),1.0)
                    ]
        }
    return size, dump_mdp(MDP), name


def get_1_medium_test_array():
    # small neutral array, offest nest
    name = "1_medium_test_array"
    size = 8
    MDP = {"nest":map_coord_to_index(size, 1, 1),
            "targets": [(map_coord_to_index(size, 7, 7),1.0)
                    ]
        }
    return size, dump_mdp(MDP), name

def get_2_medium_test_array():
    # small neutral array, offest nest
    name = "2_medium_test_array"
    size = 8
    MDP = {"nest":map_coord_to_index(size, 1, 1),
            "targets": [(map_coord_to_index(size, 4, 4),1.0),
                        (map_coord_to_index(size, 7, 7),1.0)
                    ]
        }
    return size, dump_mdp(MDP), name


def get_3_medium_test_array():
    # small neutral array, offest nest
    name = "3_medium_test_array"
    size = 8
    MDP = {"nest":map_coord_to_index(size, 1, 1),
            "targets": [(map_coord_to_index(size, 2, 4),1.0),
                        (map_coord_to_index(size, 7, 7),1.0),
                        (map_coord_to_index(size, 0, 7),1.0)
                    ]
        }
    return size, dump_mdp(MDP), name


def get_10_medium_positive_array_chittka():
    # medium positive array, offest nest, based on chitka et al 2013
    name = "10_medium_positive_array_chittka"
    size = 21
    MDP = {"size": size,
            "nest":map_coord_to_index(size, 10, 2),
            "targets": [
                    (map_coord_to_index(size, 10, 4),1.0), 
                    (map_coord_to_index(size, 13, 6),1.0),
                    (map_coord_to_index(size, 13, 9),1.0),
                    (map_coord_to_index(size, 13, 12),1.0),
                    (map_coord_to_index(size, 13, 15),1.0),
                    (map_coord_to_index(size, 10, 17),1.0),
                    (map_coord_to_index(size, 7, 15),1.0),
                    (map_coord_to_index(size, 7, 12),1.0),
                    (map_coord_to_index(size, 7, 9),1.0),
                    (map_coord_to_index(size, 7, 6),1.0)
                    ]
        }
    return size, dump_mdp(MDP), name


def get_10_medium_negative_array_chittka():
    # medium negative array, offest nest, based on chitka et al 2013
    name = "10_medium_negative_array_chittka"
    size = 21
    MDP = {"size": size,
            "nest":map_coord_to_index(size, 10, 2),
            "targets": [
                    (map_coord_to_index(size, 10, 4),1.0), 
                    (map_coord_to_index(size, 11, 7),1.0),
                    (map_coord_to_index(size, 11, 10),1.0),
                    (map_coord_to_index(size, 11, 13),1.0),
                    (map_coord_to_index(size, 11, 16),1.0),
                    (map_coord_to_index(size, 10, 19),1.0),
                    (map_coord_to_index(size, 9, 16),1.0),
                    (map_coord_to_index(size, 9, 13),1.0),
                    (map_coord_to_index(size, 9, 10),1.0),
                    (map_coord_to_index(size, 9, 7),1.0)
                    ]
        }
    return size, dump_mdp(MDP), name