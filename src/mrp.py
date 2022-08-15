import numpy as np
from json import dumps
from utils import map_coord_to_index

dump_mrp  = lambda x : dumps(x)

##########################################################################
## NOTE the order of the MRP.targets list is assumed to be the optimal 
#       trapline (which can also be reversed).
##########################################################################


def mrp_2_test_array():
    # 2 targets
    size = 8
    MRP = { 
        "name": "mrp_2_test_array",
        "size":  size,
        "optimal_sequence_length": 6,
        "nest":map_coord_to_index(size, 1, 1),
        "targets": [(map_coord_to_index(size, 4, 4),1.0),
                    (map_coord_to_index(size, 7, 7),1.0)
                ]
        }
    return dump_mrp(MRP)


def mrp_3_test_array():
    # 3 targets
    size = 8
    MRP = { 
        "name": "mrp_3_test_array",
        "size":  size,
        "optimal_sequence_length": 15,
        "nest":map_coord_to_index(size, 1, 1),
            "targets": [(map_coord_to_index(size, 2, 4),1.0),
                        (map_coord_to_index(size, 7, 7),1.0),
                        (map_coord_to_index(size, 0, 7),1.0)
                    ]
        }
    return dump_mrp(MRP)


def mrp_10_positive_array_ohashi():
    # positive array, offest nest, based on ohashi et al 2006
    size = 15
    MRP = {
            "name": "mrp_10_positive_array_ohashi",
            "size":  size,
            "optimal_sequence_length": 21,
            "nest":map_coord_to_index(size, 7, 2),
            "targets": [
                    (map_coord_to_index(size, 7, 4),1.0), 
                    (map_coord_to_index(size, 9, 5),1.0),
                    (map_coord_to_index(size, 9, 7),1.0),
                    (map_coord_to_index(size, 9, 9),1.0),
                    (map_coord_to_index(size, 9, 11),1.0),
                    (map_coord_to_index(size, 7, 12),1.0),
                    (map_coord_to_index(size, 5, 11),1.0),
                    (map_coord_to_index(size, 5, 9),1.0),
                    (map_coord_to_index(size, 5, 7),1.0),
                    (map_coord_to_index(size, 5, 5),1.0)
                    ]
        }
    return dump_mrp(MRP)


def mrp_10_negative_array_ohashi():
    # negative array, offest nest, based on ohashi et al 2006
    size = 19
    MRP = {
            "name": "mrp_10_negative_array_ohashi",
            "size":  size,
            "optimal_sequence_length": 23,
            "nest":map_coord_to_index(size, 9, 2),
            "targets": [
                    (map_coord_to_index(size, 9, 4),1.0), 
                    (map_coord_to_index(size, 10, 6),1.0),
                    (map_coord_to_index(size, 10, 9),1.0),
                    (map_coord_to_index(size, 10, 12),1.0),
                    (map_coord_to_index(size, 10, 15),1.0),
                    (map_coord_to_index(size, 9, 17),1.0),
                    (map_coord_to_index(size, 8, 15),1.0),
                    (map_coord_to_index(size, 8, 12),1.0),
                    (map_coord_to_index(size, 8, 9),1.0),
                    (map_coord_to_index(size, 8, 6),1.0)
                    ]
        }
    return dump_mrp(MRP)