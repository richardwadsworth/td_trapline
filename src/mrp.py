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
        "nest":map_coord_to_index(size, 1, 1),
            "targets": [(map_coord_to_index(size, 2, 4),1.0),
                        (map_coord_to_index(size, 7, 7),1.0),
                        (map_coord_to_index(size, 0, 7),1.0)
                    ]
        }
    return dump_mrp(MRP)


def mrp_10_positive_array_ohashi():
    # positive array, offest nest, based on ohashi et al 2006
    size = 19
    MRP = {
            "name": "mrp_10_positive_array_ohashi",
            "size":  size,
            "nest":map_coord_to_index(size, 9, 2),
            "targets": [
                    (map_coord_to_index(size, 9, 4),1.0), 
                    (map_coord_to_index(size, 12, 6),1.0),
                    (map_coord_to_index(size, 12, 8),1.0),
                    (map_coord_to_index(size, 12, 10),1.0),
                    (map_coord_to_index(size, 12, 12),1.0),
                    (map_coord_to_index(size, 9, 14),1.0),
                    (map_coord_to_index(size, 6, 12),1.0),
                    (map_coord_to_index(size, 6, 10),1.0),
                    (map_coord_to_index(size, 6, 8),1.0),
                    (map_coord_to_index(size, 6, 6),1.0)
                    ]
        }
    return dump_mrp(MRP)

def mrp_10_negative_array_ohashi():
    # negative array, offest nest, based on ohashi et al 2006
    size = 19
    MRP = {
            "name": "mrp_10_negative_array_ohashi",
            "size":  size,
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