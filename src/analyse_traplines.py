#plot learning over n experiments, showing error bars
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mlflow_utils import get_experiment_runs_data

#data, plot_rate = get_experiment_runs_data("analyse_bc764671207f4bf5b14a2f445083d0c6_10_medium_positive_array_offset")
#data, plot_rate = get_experiment_runs_data("analyse_2859cc9d8c3242918c9af22cdcb6b5d9_6_medium_positive_array_offset")

#data, plot_rate = get_experiment_runs_data("analyse_0b07230d28ed43aabe9f04aaebe1afbe_6_medium_positive_array_offset") #after MDP refactor
data, plot_rate = get_experiment_runs_data("analyse_8c76ffb6fae54f4893adfdf7804c1b7a_10_medium_positive_array_offset")

all_runs_in_experiment = data["observations"]
all_runs_done = data["done"]
MDP = data["MDP"]

total_num_samples = all_runs_in_experiment.shape[0]
num_samples_per_run = all_runs_in_experiment.shape[1]

# filter out all runs that did not return to the nest
runs_returned_to_nest = []
for i in range(total_num_samples):
    if all_runs_done[i]:
        runs_returned_to_nest.append(all_runs_in_experiment[i])

#identify the a trapline
optimal_trapline_master = [observation[0] for observation in MDP["targets"]]

# calculate the resevered MDP.  Note:  Only works for diamonds where the first target is a corner
remaining = optimal_trapline_master[1:]
remaining.reverse()
optimal_trapline_reversed_master = [optimal_trapline_master[0]] + remaining #get the optimal trapline the other direction

# optimal_trapline_master.append(MDP["nest"]) # add the nest to the trapline
# optimal_trapline_reversed_master.append(MDP["nest"]) # add the nest to the trapline

route_data = {"route":np.array(total_num_samples, dtype=object),
            "count":np.zeros(total_num_samples)}
results = pd.DataFrame(route_data)

#for each episode, find the order that the the targets where discovered in
for i in range(total_num_samples):
    run_observations = all_runs_in_experiment[i] # all the observations in a specific run.  i,e. all the episodes and their runs

    run_route_only_targets = [] # all filtered routes for this run
    for k in range(num_samples_per_run):

        episode = run_observations[k]

        #filter out all steps that are not targets.  only take the first time the target was hit (i.e. ignore when a target is revisited)
        trapline_lookup = optimal_trapline_master.copy()
        
        #get the route from the observations
        route = [observation[0] for observation in episode]

        route_only_targets = [] # the filtered route of only the targets
        for index in route:
            if index in trapline_lookup: # is this index a target?
                route_only_targets.append(index) # record the order the target was found
                trapline_lookup.remove(index) #remove the target incase it was visited again

        if len(route_only_targets)==len(optimal_trapline_master) and route[-1] == MDP["nest"]:
            #this episode found all the routes and the nest
            route_only_targets.append(MDP["nest"])

        if len(route_only_targets)>0:
            #at least one target discovered
            run_route_only_targets.append(route_only_targets)

        # if a trapline was created in this run, then the latter episodes should visit the targets in the same order

    count = pd.Series(run_route_only_targets).value_counts().sort_values(ascending=False)
    
    most_popular_route = count.index[0] # we ordered by count of routes descending
    
    
    results["route"][i] = count.index[0]
    results["count"][i] = count[0]
    
    # res = (most_popular_route == optimal_trapline_master) or (most_popular_route == optimal_trapline_reversed_master)
    # if res:
    #     #this is an optimal trapline 
    #     pass
    # else:
    #     pass


count3 = pd.Series(results["route"]).value_counts().sort_values(ascending=False)

x = [str(x) for x in count3.index.to_list()]
fig, ax = plt.subplots(1,1)
ax.bar(x, count3)
ax.set_xticklabels(x, rotation = 90)
fig.tight_layout()
plt.grid()
plt.show()
1==1

    

1==1


