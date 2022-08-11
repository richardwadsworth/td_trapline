import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns

from mlflow_utils import get_experiment_runs_data
from utils import get_sliding_window_sequence
from trapline import get_optimal_trapline_for_diamond_array, get_routes_similarity, get_valid_target_sequence_from_route, RouteType
from plots import plot_route, plot_trapline_distribution
 

#experiment_name = "10_medium_positive_array_chittka_hyperparameter_gridsearch" # 10 positive chittka
experiment_name = "10_medium_negative_array_chittka_hyperparameter_gridsearch" #10 negative chittka


data, plot_rate = get_experiment_runs_data(experiment_name) 

metrics = data['metrics']

score_softmax = [d['score_softmax'] for d in metrics]
score_softmax_5 = [d['score_softmax_last_05'] for d in metrics]
score_softmax_20 = [d['score_softmax_last_20'] for d in metrics]

import re, seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# generate data
n = 200

# axes instance
fig = plt.figure()
ax = plt.axes(projection='3d')
fig.add_axes(ax)

# get colormap from seaborn
cmap = ListedColormap(sns.color_palette("crest", 256).as_hex())

# plot
sc = ax.scatter(score_softmax, score_softmax_5, score_softmax_20, s=40, c=score_softmax, cmap=cmap, marker='o', alpha=1)
ax.set_xlabel('Final Performance Score')
ax.set_ylabel('Mean Last 5 Performance Scores')
ax.set_zlabel('Mean Last 20 Performance Scores')

fig.suptitle("Hyperparameter Gridsearch - Maximise Reward Performance")
ax.set_title(experiment_name, fontsize=10)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()
#plt.savefig("scatter_hue", bbox_inches='tight')
