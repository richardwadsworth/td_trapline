'''
 Script to analyse the results of a parameter sweep and out put the parameter set with the
 highest average cumulative reward

 Call this script by passing the in the grid search experiment name as the argument.

 e.g. 
  python ./src/analyse_hyperparameter_sweep.py mrp_10_positive_array_ohashi_gs
 
'''
import argparse
from unittest import result
import pandas as pd
import hashlib
import json
import os
from os.path import exists
import seaborn as sns
sns.set_theme(style="whitegrid")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mlflow_utils import get_experiment_runs_data
import seaborn as sns
 
def main():

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    args = parser.parse_args()
    
    artifact_path = "artifacts"

    data = get_experiment_runs_data(args.experiment_name) 

    metrics = data['metrics']
    params = data['params']
    run_ids = data['run_ids']

    score_softmax = [d['score_softmax'] for d in metrics]
    score_softmax_5 = [d['score_softmax_last_05'] for d in metrics]
    score_softmax_20 = [d['score_softmax_last_20'] for d in metrics]

  
    hash_params = lambda x : hashlib.md5(json.dumps(x).encode('utf-8')).hexdigest()

    df = pd.DataFrame()
    df['params'] = params
    df['run_ids'] = run_ids
    df['params_hash'] = np.array([hash_params(d) for d in params])
    #df['score_softmax'] = np.array(score_softmax)
    df['score_softmax_last_05'] = np.array(score_softmax_5)
    df['score_softmax_last_20'] = np.array(score_softmax_20)

    # get the mean scores for each set params
    #score_softmax_mean =df.groupby(by=['params_hash'])['score_softmax'].mean()
    score_softmax_5_mean = df.groupby(by=['params_hash'])['score_softmax_last_05'].mean()
    score_softmax_20_mean = df.groupby(by=['params_hash'])['score_softmax_last_20'].mean()

    df_mean =  pd.DataFrame()
    df_mean.index = score_softmax_5_mean.index

    # find the first run_id in the set of parameters for the param hash
    df_mean['run_id'] = [df.loc[df.params_hash==index]['run_ids'].iloc[0] for index in df_mean.index]
    df_mean['score_softmax_5_mean'] = score_softmax_5_mean.values
    df_mean['score_softmax_20_mean'] = score_softmax_20_mean.values

    # remove params hash from output
    df_mean = df_mean.reset_index().drop(['params_hash'], axis=1)

    sort1 = df_mean.sort_values(['score_softmax_20_mean', 'score_softmax_5_mean'], ascending=[False, False])
    sort2 = df_mean.sort_values(['score_softmax_5_mean', 'score_softmax_20_mean'], ascending=[False, False])
    
    def output_results(f, r):
        print(r)
        f.write(r)

    #save results to file
    filepath = os.path.join(artifact_path, 'analyse_' + args.experiment_name + '_grid_search_results.csv')
    if exists(filepath):
        os.remove(filepath)

    with open(filepath, "w+") as file:

        output_results(file, "Sorted by last 20 then last 5")
        output_results(file, sort1.to_string())
        output_results(file, "\n\n")
        output_results(file, "Sorted by last 5 then last 20")
        output_results(file, sort2.to_string())

    fig, ax  = plt.subplots()
    
    ax.scatter(score_softmax_5_mean, score_softmax_20_mean, marker='o', alpha=1)

    ax.set_xlabel('Mean Last 5 Performance Scores')
    ax.set_ylabel('Mean Last 20 Performance Scores')
    ax.set_ylim(-7,10)
    ax.set_xlim(-7,10)

    fig.suptitle("Hyperparameter Gridsearch - Maximise Reward Performance")
    ax.set_title(args.experiment_name, fontsize=10)

    filepath = os.path.join(artifact_path, 'analyse_' + args.experiment_name + '_grid_search_scatter.png')
    plt.savefig(filepath, bbox_inches='tight', format='png')

    plt.show()
    

if __name__ == "__main__":
   main()