import numpy as np

def map_coord_to_index(size, x, y):
    return (y*size)+x

def map_index_to_coord(size, index): 
        x= index%size
        y= int(np.floor(index/size))
        return [x, y]

def sliding_window(window_size, sequence):
    sliding_sequence = []
    for i in range(len(sequence) - window_size + 1):
        sliding_sequence.append(sequence[i: i + window_size])
    
    return sliding_sequence

def get_sliding_window_sequence(sliding_window_size, num_routes, stable_point_ratio):

    # get the latter episode samples using the stable_point as the cut off point
    num_episode_samples_to_sample = int(num_routes*stable_point_ratio)
    episode_samples_to_sample = np.arange(num_routes)[-num_episode_samples_to_sample:]

    # get a sliding window of adjacent routes
    sample_sequence = sliding_window(sliding_window_size, episode_samples_to_sample)

    return sample_sequence

