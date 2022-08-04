from math import ceil

def map_index_to_coord(size, index): 
        getRow = lambda i: ceil((i+1)/size)-1
        getCol = lambda i: i%size
        x = getCol(index)
        y=  getRow(index)
        return [x, y]