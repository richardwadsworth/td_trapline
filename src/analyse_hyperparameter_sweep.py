import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns

from mlflow_utils import get_experiment_runs_data
 

experiment_name = "mrp_10_negative_array_ohashi_gs" # 10 negative ohashi
#experiment_name = "10_medium_positive_array_ohashi_gs" #10 positive ohashi

data, plot_rate = get_experiment_runs_data(experiment_name) 

metrics = data['metrics']
params = data['params']
run_ids = data['run_ids']

score_softmax = [d['score_softmax'] for d in metrics]
score_softmax_5 = [d['score_softmax_last_05'] for d in metrics]
score_softmax_20 = [d['score_softmax_last_20'] for d in metrics]

import hashlib
import json
hash_params = lambda x : hashlib.md5(json.dumps(x).encode('utf-8')).hexdigest()


df = pd.DataFrame()
df['params'] = params
df['run_ids'] = run_ids
df['params_hash'] = np.array([hash_params(d) for d in params])
df['score_softmax'] = np.array(score_softmax)
df['score_softmax_last_05'] = np.array(score_softmax_5)
df['score_softmax_last_20'] = np.array(score_softmax_20)

# get the mean scores for each set params
score_softmax_mean =df.groupby(by=['params_hash'])['score_softmax'].mean()
score_softmax_5_mean = df.groupby(by=['params_hash'])['score_softmax_last_05'].mean()
score_softmax_20_mean = df.groupby(by=['params_hash'])['score_softmax_last_20'].mean()

#print out the best average score_softmax_20_mean

def get_best_average(df, column, column_name):

    mean_max =  column.max()
    param_hash_id_column_max =  column.idxmax()
    run_id = df.loc[df.params_hash==param_hash_id_column_max]['run_ids'].iloc[0] #all param sets will be the same because we have grouped by param set, so just take the first row
    param_set = dict(df[df['run_ids']==run_id]['params'])
    print()
    print('Best average {}: {}'.format(column_name, mean_max))
    print('Run ID: {}'.format(run_id))
    print('Param Hash: {}'.format(param_hash_id_column_max))
    print('Param set: {}'.format(param_set))

    return run_id, mean_max

score_softmax_mean_param_set_run_id, score_softmax_mean_param_set_score = get_best_average(df, score_softmax_mean, "score_softmax_mean")
score_softmax_5_mean_param_set_run_id, score_softmax_5_mean_param_set_score = get_best_average(df, score_softmax_5_mean, "score_softmax_5_mean")
score_softmax_20_mean_param_set_run_id, score_softmax_20_mean_param_set_score = get_best_average(df, score_softmax_20_mean, "score_softmax_20_mean")

print("Do the param set hashes match.  If so one set of config generated the best overall scores.")

print("Summary")
print("score_softmax_mean: Run ID: {} Score: {}".format(score_softmax_mean_param_set_run_id, score_softmax_mean_param_set_score))
print("score_softmax_5_mean: Run ID: {} Score: {}".format(score_softmax_5_mean_param_set_run_id, score_softmax_5_mean_param_set_score))
print("score_softmax_20_mean: Run ID: {} Score: {}".format(score_softmax_20_mean_param_set_run_id, score_softmax_20_mean_param_set_score))

import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# axes instance
fig = plt.figure()
ax = plt.axes(projection='3d')
fig.add_axes(ax)

# get colormap from seaborn
cmap = ListedColormap(sns.color_palette("crest", 256).as_hex())

# plot
sc = ax.scatter(score_softmax_mean, score_softmax_5_mean, score_softmax_20_mean, s=40, c=score_softmax_mean, cmap=cmap, marker='o', alpha=1)
ax.set_xlabel('Final Performance Score')
ax.set_ylabel('Mean Last 5 Performance Scores')
ax.set_zlabel('Mean Last 20 Performance Scores')

fig.suptitle("Hyperparameter Gridsearch - Maximise Reward Performance")
ax.set_title(experiment_name, fontsize=10)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()
#plt.savefig("scatter_hue", bbox_inches='tight')
