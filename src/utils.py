import numpy as np

def map_coord_to_index(size, x, y):
    return (y*size)+x

def map_index_to_coord(size, index): 
        x= index%size
        y= int(np.floor(index/size))
        return [x, y]

def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

def sliding_window(window_size, sequence):
    sliding_sequence = []
    for i in range(len(sequence) - window_size + 1):
        sliding_sequence.append(sequence[i: i + window_size])
    
    return sliding_sequence

def get_sliding_window_sequence(sliding_window_size, num_routes):

    episode_samples_to_sample = np.arange(num_routes)

    # get a sliding window of adjacent routes
    sample_sequence = sliding_window(sliding_window_size, episode_samples_to_sample)

    return sample_sequence

