import numpy as np

def map_coord_to_index(size, x, y):
    return (y*size)+x

def map_index_to_coord(size, index): 
        x= index%size
        y= int(np.floor(index/size))
        return [x, y]

