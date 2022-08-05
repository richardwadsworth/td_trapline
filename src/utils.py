import numpy as np

def map_index_to_coord(size, index): 
        x= index%size
        y= int(np.floor(index/size))
        return [x, y]

