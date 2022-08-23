import numpy as np

def map_coord_to_index(size, x, y):
    return (y*size)+x

def map_index_to_coord(size, index): 
        x= index%size
        y= int(np.floor(index/size))
        return [x, y]


def moving_average(x, w):
    """
    create a moving average by convolving a data graph

    Args:
        x: raw data
        w: window size
    """
    return np.convolve(x, np.ones(w), 'valid') / w

def sliding_window(window_size, sequence):
    """
    Create a sliding window from a sequence
    """
    sliding_sequence = []
    for i in range(len(sequence) - window_size + 1):
        sliding_sequence.append(sequence[i: i + window_size])
    
    return sliding_sequence

def get_sliding_window_sequence(sliding_window_size, num_routes):
    """
    create a sliding window based on a number of routes

    Args:
        sliding_window_size: size of window
        num_routes: number of routes

    Returns:
        a sequence of route Id tuples
    """

    episode_samples_to_sample = np.arange(num_routes)

    # get a sliding window of adjacent routes
    sample_sequence = sliding_window(sliding_window_size, episode_samples_to_sample)

    return sample_sequence

