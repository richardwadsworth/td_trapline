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
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mlflow_utils import get_experiment_runs_data
 
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
    df['score_softmax'] = np.array(score_softmax)
    df['score_softmax_last_05'] = np.array(score_softmax_5)
    df['score_softmax_last_20'] = np.array(score_softmax_20)

    # get the mean scores for each set params
    score_softmax_mean =df.groupby(by=['params_hash'])['score_softmax'].mean()
    score_softmax_5_mean = df.groupby(by=['params_hash'])['score_softmax_last_05'].mean()
    score_softmax_20_mean = df.groupby(by=['params_hash'])['score_softmax_last_20'].mean()

    #print out the best average score_softmax_20_mean

    def output_results(f, r):
        print(r)
        f.write(r)

    def get_best_average(file, df, column, column_name):

        mean_max =  column.max()
        param_hash_id_column_max =  column.idxmax()

        #all param sets will be the same because we have grouped by the winning param set, so just take the last row
        run_id = df.loc[df.params_hash==param_hash_id_column_max]['run_ids'].iloc[0] 
        param_set = dict(df[df['run_ids']==run_id]['params'])
        
        results = ""
        results += 'Best average {}: {}'.format(column_name, mean_max) + '\n'
        results += 'Run ID: {}'.format(run_id) + '\n'
        results += 'Param Hash: {}'.format(param_hash_id_column_max) + '\n'
        results += 'Param set: {}'.format(param_set) + '\n'

        output_results(file, results)

        return run_id, mean_max

    #save results to file
    filepath = os.path.join(artifact_path, 'analyse_' + args.experiment_name + '_grid_search_results.csv')
    if exists(filepath):
        os.remove(filepath)

    with open(filepath, "w+") as file:

        score_softmax_mean_param_set_run_id, score_softmax_mean_param_set_score = get_best_average(file, df, score_softmax_mean, "score_softmax_mean")
        score_softmax_5_mean_param_set_run_id, score_softmax_5_mean_param_set_score = get_best_average(file, df, score_softmax_5_mean, "score_softmax_5_mean")
        score_softmax_20_mean_param_set_run_id, score_softmax_20_mean_param_set_score = get_best_average(file, df, score_softmax_20_mean, "score_softmax_20_mean")
        
        results=""
        results+="Do the param set hashes match.  If so one set of config generated the best overall scores.\n"
        results+="Summary\n"
        results+="score_softmax_mean: Run ID: {} Score: {}".format(score_softmax_mean_param_set_run_id, score_softmax_mean_param_set_score) + "\n"
        results+="score_softmax_5_mean: Run ID: {} Score: {}".format(score_softmax_5_mean_param_set_run_id, score_softmax_5_mean_param_set_score) + "\n"
        results+="score_softmax_20_mean: Run ID: {} Score: {}".format(score_softmax_20_mean_param_set_run_id, score_softmax_20_mean_param_set_score)
        output_results(file, results)


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
    ax.set_title(args.experiment_name, fontsize=10)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    filepath = os.path.join(artifact_path, 'analyse_' + args.experiment_name + '_grid_search_scatter.png')
    plt.savefig(filepath, bbox_inches='tight', format='png')

    plt.show()
    

if __name__ == "__main__":
   main()