import contextlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
import seaborn as sns
from json import loads
import os

from enum import Enum
from IPython.display import display, clear_output
from sklearn.preprocessing import Normalizer
from timeColouredPlots import doColourVaryingPlot2d
from trapline import TargetSequenceType
from utils import map_index_to_coord
from foraging_agent import ActionType
from utils import map_index_to_coord

# Plotting verbosity 
class PlotType(Enum):
    NoPlots = 0 # do not display any plots
    Minimal = 1 # only display plot if threshold exceeded
    Partial = 2 # display plot at end of training
    Full = 3 # display and update plot throughout trainingx

def remove_axis_ticks(ax):
        ax.tick_params(
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,        # ticks along the left edge are off
            right=False,        # ticks along the right edge are off
            labelbottom=False,
            labelleft=False) # labels along the bottom edge are off

def initialise_plots(env):

    # set up the subplots for visualisation
    fig1, axs = plt.subplots(2,3, figsize=(11,7))
    sns.set_theme(style="whitegrid")
    
    ax1, ax2, ax3, ax4, ax5, ax6 = axs.ravel()
    remove_axis_ticks(ax1)
    remove_axis_ticks(ax2)
    remove_axis_ticks(ax3)
    remove_axis_ticks(ax5)
    remove_axis_ticks(ax6)

    # create coordinate lookup table
    xs_coordinate_map, ys_coordinate_map = [], []
    for index in range(env.observation_space[0].n):
        x, y = map_index_to_coord(env.size, index)
        xs_coordinate_map.append(x)
        ys_coordinate_map.append(y)

    xs_target, ys_target = [],[]
    for index in env.target_indices:
        xs_target.append(xs_coordinate_map[index])
        ys_target.append(ys_coordinate_map[index])
    
    return (fig1, ax1, ax2, ax3, ax4, ax5, ax6, xs_coordinate_map, ys_coordinate_map, xs_target, ys_target)


# %%
def plotActionStateQuiver(env, q, fig1, ax1, ax2, xs_target, ys_target):

    # %%
    def resolveActionState(actionState):

        #W, S, E, N
        vertical = actionState[ActionType.NORTH.value] - actionState[ActionType.SOUTH.value] 
        horizontal = actionState[ActionType.EAST.value] - actionState[ActionType.WEST.value] 

        return horizontal, vertical

    #get average action state values across all possible actions.  i.e. get a 2d slice of the 3d matrix
    q_mean = np.mean(q, axis=(0))

    policyFound = [resolveActionState(q_mean[x,:]) for x in range(env.observation_space[0].n)]
        
    i = np.arange(0, env.size) #rows
    j = np.arange(0, env.size) #colums

    ii, jj = np.meshgrid(i,j)#, indexing='ij')

    # print("row indices:\n{}\n".format(ii))
    # print("column indices:\n{}".format(jj))

    U = np.reshape([i[0] for i in policyFound], (env.size, env.size))
    V = np.reshape([i[1] for i in policyFound], (env.size, env.size))

    # Normalize the arrows:
    U_norm = U/np.max(np.abs(U)) 
    V_norm = V/np.max(np.abs(V))

    ax1.cla()
    ax2.cla()

    nest_x, nest_y = map_index_to_coord(env.size, env.nest_index)
    ax1.scatter(nest_x,nest_y, c='brown', s=50, marker='^') #origin
    ax2.scatter(nest_x,nest_y, c='brown', s=50, marker='^') #origin
    
    ax1.scatter(xs_target,ys_target, c='r', s=50, marker='o') #goal
    ax2.scatter(xs_target,ys_target, c='r', s=50, marker='o') #goal
    
    QP = ax1.quiver(ii,jj, U_norm, V_norm, scale=10)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    ax1.title.set_text('Normalised Quiverplot')


    QP = ax2.quiver(ii,jj, U, V)
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    ax2.title.set_text('Raw')
    
    with contextlib.redirect_stdout(None):
        display(fig1)    
    clear_output(wait = True)
    plt.pause(0.0000000001)

# %%
def plotAgentPath(env, fig1, ax3, ax4, xs_coordinate_map, ys_coordinate_map, xs_target, ys_target):
    
    #set up plot
    if ax3.get_title() == "":
        # ax4.scatter(xs_target,ys_target, c='brown', s=20, marker='o') #goal
        showbar = True
        
        # ax4.set_title("All Agent path")
    else:
        showbar=False

    ax3.cla()

    nest_x, nest_y = map_index_to_coord(env.size, env.nest_index)
    ax3.scatter(nest_x,nest_y, c='brown', s=50, marker='^') #origin
    
    ax3.scatter(xs_target,ys_target, c='brown', s=50, marker='o') #goal
    ax3.set_title("Agent path")
    ax3.grid()

    xs, ys = [], []
    for index, orientation in env.observations:
        xs.append(xs_coordinate_map[index])
        ys.append(ys_coordinate_map[index])

    ts = np.arange(0, len(xs))

    doColourVaryingPlot2d(xs, ys, ts, fig1, ax3, map='plasma', showBar=showbar, barlabel='Step')  # only draw colorbar once
    
    # fix plot axis proportions to equal
    ax3.set_aspect('equal')
    ax3.set_xlim([-1, env.size])
    ax3.set_ylim([0-1, env.size])
    ax3.invert_yaxis()

    with contextlib.redirect_stdout(None):
        display(fig1) 

    clear_output(wait = True)
    plt.pause(0.0000000001)


def plot_performance(fig1, ax, episodes, steps, performance, sample_rate):

    
    ax.plot(sample_rate*np.arange(episodes//sample_rate), performance)
    ax.set_xlabel("Episodes")
    ax.set_title("Learning progress for SARSA")
    ax.set_ylabel("Average reward of an Episode")
    ax.grid()
    fig1.tight_layout()

    with contextlib.redirect_stdout(None):
        display(fig1)    
    clear_output(wait = True)
    plt.pause(0.0000000001)


def plot_traffic_noise(env, fig, ax, xs_coordinate_map, ys_coordinate_map, xs_target, ys_target, data, title, sigma = 0.05, alpha=0.2, linewidth=1.5):

    def plot(x, y):
        x_ = x + np.random.normal(0,sigma,len(x))
        y_ = y + np.random.normal(0,sigma,len(y))
        ax.plot(x_,y_, color="black", alpha=alpha, linewidth=linewidth)

    for observations in data:
        #extract the index data from the observations
        coords = [[xs_coordinate_map[observation[0]],ys_coordinate_map[observation[0]]] for observation in observations]
        x = [coord[0] for coord in coords]
        y = [coord[1] for coord in coords]
        plot(x, y)

    nest_x, nest_y = map_index_to_coord(env.size, env.nest_index)
    ax.scatter(nest_x,nest_y, c='brown', s=50, marker='^') #origin
    ax.scatter(xs_target,ys_target, c='r', s=50, marker='o') #goal

    ax.set_xlim(-1,env.size)
    ax.set_ylim(-1,env.size)
    ax.set_title("Traffic plot of Agent under {}".format(title))
    
    ax.invert_yaxis()

    with contextlib.redirect_stdout(None):
        display(fig)    
    clear_output(wait = True)
    plt.pause(0.0000000001)

    
    
        


def plot_traffic_greyscale(env, fig, ax, xs_target, ys_target, data, title):
    
    def create_segment(route):
        points = np.array(route).reshape(-1,1,2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    s = []
    for observations in data:
        #extract the index data from the observations
        route = [map_index_to_coord(env.size,x[0]) for x in observations]
        s += create_segment(route).tolist()

    import pandas as pd
    count = pd.Series(s).value_counts().sort_values(ascending=True)
     
    _s = count.index
    _width = count.values

    name = "binary"
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # Use a boundary norm instead
    cmap = plt.get_cmap(name) # ListedColormap(['r', 'g'])

    X =  _width
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (230 - 20) + 20

    boundaries = np.arange(0,256, 1)
    norm = BoundaryNorm(boundaries, cmap.N)
    lc = LineCollection(_s, cmap=cmap, norm=norm)
    lc.set_array(X_scaled)
    lc.set_linewidth(2)

    
    ax.set_xlim(-1,env.size)
    ax.set_ylim(-1,env.size)
    ax.set_title("Traffic plot of Agent under {}".format(title))

    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    # a.grid()

    nest_x, nest_y = map_index_to_coord(env.size, env.nest_index)
    ax.scatter(nest_x,nest_y, c='brown', s=50, marker='^') #origin
    
    ax.scatter(xs_target,ys_target, c='r', s=50, marker='o') #goal
    
    ax.invert_yaxis()

    with contextlib.redirect_stdout(None):
        display(fig)    
    clear_output(wait = True)
    plt.pause(0.0000000001)

def plot_target_sequence(experiment_name, artifact_path, filename_suffix, fig, ax, size, nest, targets, target_sequence, target_sequence_type, subtitle):
    
    remove_axis_ticks(ax)

    target_sequence_coords = [map_index_to_coord(size, index) for index in target_sequence]
    target_coords = [map_index_to_coord(size, index) for index in targets]
    
    from utils import get_sliding_window_sequence
    sequence = get_sliding_window_sequence(2,len(target_sequence_coords))

    ax.set_xlim(0,size)
    ax.set_ylim(0,size)

    nest_x, nest_y = map_index_to_coord(size, nest)    
    ax.scatter(nest_x,nest_y, c='brown', s=50, marker='^') #origin

    xs_target = [coord[0] for coord in target_coords]
    ys_target = [coord[1] for coord in target_coords]
    ax.scatter(xs_target,ys_target, c='r', s=50, marker='o') #goal
    
    if target_sequence_type == TargetSequenceType.Optimal:
        color = 'g' 
        linestyle = 'solid' 
        label= 'Optimal' 
    
    elif target_sequence_type == TargetSequenceType.SubOptimal:

        color = 'b' 
        linestyle = 'dashed'
        label= 'Sub Optimal'
    else:
        color = 'grey' 
        linestyle = 'dashed'
        label= 'Incomplete'

    for window in sequence:
        xy1 = target_sequence_coords[window[0]]
        xy2 = target_sequence_coords[window[1]]
        t = ax.annotate("", xy=xy2, xytext=xy1,
                arrowprops=dict(color=color, arrowstyle="->", lw=2.5, linestyle=linestyle))
    
    ax.plot([0],[0], label=label, color=color, lw=2.5, linestyle=linestyle)
    ax.legend(loc='upper right', prop={'size': 8})
    
    ax.set_title("Trapline " + subtitle)
    fig.tight_layout()
    ax.invert_yaxis()
    
    with contextlib.redirect_stdout(None):
        display(fig)    
    clear_output(wait = True)

    filepath = os.path.join(artifact_path, experiment_name + filename_suffix)
    fig.savefig(filepath + '.png')


    plt.pause(0.0000000001)

def plot_traplines_where_second_is_in_unexpected_order(experiment_name, artifact_path, MRP, df_target_sequence_data_for_experiment, optimal_trapline):
        '''
        search all sequences to find out how many sequences have the nearest and second nearest targets as the first and second discovered targets

        note this function is only value for circular/diamond arrays with one target nearest the nest
        '''

        df = df_target_sequence_data_for_experiment # alias
        sequences = df["target_sequence"]

        # strip out invalid trapline
        sequences = [s for s in sequences if s != []]

        first_nearest_targets = [optimal_trapline[0]]
        second_nearest_targets = [optimal_trapline[1]] + [optimal_trapline[-1]]
        sequence_nearest_first = [sequence for sequence in sequences if sequence[0] in first_nearest_targets]
        sequence_nearest_second = [sequence for sequence in sequence_nearest_first if  sequence[1] in second_nearest_targets]

        print("Number of traplines where first target is not first nearest to the nest: {}".format(len(sequences)- len(sequence_nearest_first)))
        print("Number of traplines where second target is not second nearest to the nest: {}".format(len(sequences)- len(sequence_nearest_second)))

        import seaborn as sns
        from plots import plot_target_sequence

        if len(sequences)- len(sequence_nearest_second)>0:
            unexpected =  [sequence for sequence in sequence_nearest_first if  sequence[1] not in second_nearest_targets]
            df_unexpected = df.loc[df["target_sequence"].isin(unexpected)]


            plot_size = int(np.ceil(np.sqrt(len(df_unexpected))))
            fig, axs = plt.subplots(plot_size, plot_size, figsize=(plot_size*3, plot_size*3))
            sns.set_theme(style="whitegrid")
            
            fig.suptitle("Traplines where second target is not second nearest\n\n" + experiment_name, fontsize=10)

            axs = np.array(axs).reshape(-1)

            for i, ax in enumerate(axs):

                if i >= len(df_unexpected):
                    break
                # Convert string representation of target sequence to list using json
                target_sequence = df_unexpected["target_sequence_including_nest"][df_unexpected.index[i]]
                target_sequence_id= df_unexpected['target-sequence-id'][df_unexpected.index[i]]
                target_sequence_type= df_unexpected['target_sequence_type'][df_unexpected.index[i]]
                ax.grid(visible=False) # turn off the grid
                plot_target_sequence(experiment_name, artifact_path, '_trapline_id_unexpected', fig, ax, int(MRP["size"]),MRP["nest"], optimal_trapline, target_sequence, target_sequence_type, str(target_sequence_id))


def plot_trapline_distribution(experiment_name, artifact_path, num_runs_in_experiment, MRP, df_target_sequence_data, optimal_trapline):

    LABEL_INVALID_TRAPLINE = 'Invalid Trapline'

    df = df_target_sequence_data # alias for ease of reading

    # build x-axis labels
    counter = 1
    x_axis_label = [] #used for labelling x-axis
    target_sequence_id = [] # id assigned to target sequence to enable lookup of routing
    for i, r in enumerate(df["target_sequence_including_nest"]):
        if r == []:
            x_axis_label.append(LABEL_INVALID_TRAPLINE)
            target_sequence_id.append(LABEL_INVALID_TRAPLINE)
        else:
            if counter%2==0: # only display a label every 2 to avoid over crowding
                x_axis_label.append(str(counter))
            else:
                x_axis_label.append('')

            target_sequence_id.append(str(counter))
            counter += 1

    df['target-sequence-id'] = target_sequence_id

    # determine the optimality of each target sequence
    def get_target_sequence_type(target_sequence_length):
        if target_sequence_length == 1.0: # since the length is normalised we know the optimal length is 1
            return TargetSequenceType.Optimal
        elif target_sequence_length == 0:
            return TargetSequenceType.Incomplete
        else:
            return TargetSequenceType.SubOptimal  # all targets found, but not in the optimal order

    df['target_sequence_type'] = [get_target_sequence_type(target_sequence) for target_sequence in df["sequence_manhattan_length"]]

    #plot bar chart
    fig1, ax = plt.subplots(1,1, figsize=(12,5))
    sns.set_theme(style="whitegrid")

    bar_list = ax.bar(df['target-sequence-id'], df["target_sequence_count"], edgecolor = "black") # plot the bar chart

    filepath = os.path.join(artifact_path, experiment_name + '_trapline_distribution')
    df.to_csv(filepath + ".csv")

    ax.set_xlabel('Trapline ID')
    
    ax.set_yscale('log')
    ax.set_ylim(0, num_runs_in_experiment)
    ax.set_ylabel('Logarithmic Count of Trapline')

    fig1.suptitle("Trapline Distribution")
    ax.set_title(experiment_name, fontsize=10)

    # highlight the optimal target sequences, if present in the results
    for i in range(len(df)):
        target_sequence_type = df['target_sequence_type'][i]
        if target_sequence_type == TargetSequenceType.Incomplete:
            bar_list[i].set_color('grey')
        elif target_sequence_type == TargetSequenceType.Optimal:
            bar_list[i].set_color('g')
        elif target_sequence_type == TargetSequenceType.SubOptimal:
            bar_list[i].set_color('b')

    ax.set_xticklabels(x_axis_label, rotation = 90)

    ax.grid(visible=True)
    
    fig1.tight_layout()
    
    
    fig1.savefig(filepath + '.png')


    # drop the target sequence count with no discernable target based route found
    df = df.drop(df[df['target-sequence-id'] == LABEL_INVALID_TRAPLINE].index)

    # if there are more than 9 target sequences, choose the optimal and 7 random
    MAX_NUM_ROUTE_PLOTS = 9
    if len(df)>MAX_NUM_ROUTE_PLOTS:

        df_first_two_in_list = df.iloc[[0,1]]

        #find index of optimal target sequences
        df_optimal = df.drop(df[df['target_sequence_type'] != TargetSequenceType.Optimal].index)
        
        #drop from the first two if they are in fact the optimal target sequences
        df_first_two_in_list = df_first_two_in_list.drop(df_optimal.index, errors='ignore')

        remaining = MAX_NUM_ROUTE_PLOTS - (len(df_first_two_in_list) + len(df_optimal))

        # get suboptimal target sequences and select a random selection for display
        df_suboptimal = df.drop(df[df['target_sequence_type'] != TargetSequenceType.SubOptimal].index)
        suboptimal_indexes_for_plot = np.random.choice(df_suboptimal.index, size=remaining, replace=False)
        df_suboptimal_for_plot= df_suboptimal.loc[suboptimal_indexes_for_plot]
        df_for_plotting = pd.concat([df_first_two_in_list, df_optimal, df_suboptimal_for_plot])
        df_for_plotting = df_for_plotting.sort_index()
     
    plot_size = int(np.ceil(np.sqrt(len(df_for_plotting))))
    fig2, axs = plt.subplots(plot_size, plot_size, figsize=(plot_size*3, plot_size*3))
    sns.set_theme(style="whitegrid")
    
    fig2.suptitle("Trapline ID Lookup for Trapline Distribution\n\n" + experiment_name, fontsize=10)

    axs = np.array(axs).reshape(-1)

    for i, ax in enumerate(axs):

        if i >= len(df_for_plotting):
            break
        # Convert string representation of target sequence to list using json
        target_sequence = df_for_plotting["target_sequence_including_nest"][df_for_plotting.index[i]]
        target_sequence_id= df_for_plotting['target-sequence-id'][df_for_plotting.index[i]]
        target_sequence_type= df_for_plotting['target_sequence_type'][df_for_plotting.index[i]]
        ax.grid(visible=False) # turn off the grid
        plot_target_sequence(experiment_name, artifact_path, "_trapline_id_lookup", fig2, ax, int(MRP["size"]),MRP["nest"], optimal_trapline, target_sequence, target_sequence_type, str(target_sequence_id))

    
    
    plot_traplines_where_second_is_in_unexpected_order(experiment_name, artifact_path, MRP,  df, optimal_trapline)
    

    plt.subplots_adjust(left=0.1, right=0.9, top=0.86, bottom=0.15)
    plt.pause(0.00000000001)


    

def plot_similarity_scores(experiment_name, artifact_path, sample_rate, route_similarity_score, route_similarity_score_rate_of_change):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5))
    sns.set_theme(style="whitegrid")

    for similarity_score, similarity_score_prime in zip(route_similarity_score, route_similarity_score_rate_of_change):

        xs = np.arange(0, len(similarity_score)) * sample_rate
       
        alpha = 0.7
        ax1.plot(xs, similarity_score, alpha=alpha)
        ax2.plot(xs, similarity_score_prime, alpha=alpha)

    
    # calculate the mean C score across all runs
    df = pd.DataFrame([x for x in route_similarity_score])
    similarity_score_mean = df.mean() 
    xs = np.arange(0, len(similarity_score_mean)) * sample_rate
    alpha = 1
    ax1.plot(xs, similarity_score_mean, alpha=alpha, lw=2, color='black', label="Mean C score")
    ax1.legend(loc='upper right')

    df = pd.DataFrame([x for x in route_similarity_score_rate_of_change])
    similarity_score_prime_mean = df.mean() 
    xs = np.arange(0, len(similarity_score_prime_mean)) * sample_rate
    alpha = 1
    ax2.plot(xs, similarity_score_prime_mean, alpha=alpha, lw=2, color='black', label="Mean C score rate of change")
    ax2.legend(loc='upper right')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('(Smoothed) C Score')
    ax1.set_ylim(0, 1600)
    ax1.set_title("C Score per episode for all runs", fontsize=10)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Rate of change of C Score')
    ax2.set_ylim(-400, 400)
    ax2.set_title("Rate of change of C Score per episode for all runs", fontsize=10)


    fig.suptitle("Route C Score by Episode\n" + experiment_name, fontsize=12)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.83, bottom=0.15)

    filepath = os.path.join(artifact_path, experiment_name + '_route_similarity_score')
    fig.savefig(filepath + '.png')

    plt.pause(0.00000000001)

def plot_stability_distribution(experiment_name, artifact_path, sample_rate, stability_threshold, route_similarity_score):

    fig, ax = plt.subplots()
    sns.set_theme(style="whitegrid")

    route_similarity_score = np.array([index for index in route_similarity_score if index!=-1]) * sample_rate # scale up from the plot (sample) rate

    sns.histplot(route_similarity_score, bins=20, ax=ax, edgecolor = "black")
    
    ax.xaxis.get_ticklocs(minor=True)
    ax.minorticks_on()

    ax.set_xlabel('Episode that route stabilites to a trapline')
    ax.set_xlim(0, 200)
    
    ax.set_yscale('log')
    ax.set_ylim(0, 600)
    ax.set_ylabel('Logarithmic Count of Traplines')

    fig.suptitle("Trapline Stability Point Distribution")
    ax.set_title(experiment_name, fontsize=10)

    
    ax.grid(visible=True)


    plt.subplots_adjust(left=0.1, right=0.9, top=0.83, bottom=0.15)

    filepath = os.path.join(artifact_path, experiment_name + '_stability')
    fig.savefig(filepath + '.png')

    plt.pause(0.00000000001)


def plot_target_sequence_length_distribution(experiment_name, artifact_path, num_runs_in_experiment, df_target_sequence_data):

    df = df_target_sequence_data.groupby(['sequence_manhattan_length']).sum()
    
    filepath = os.path.join(artifact_path, experiment_name + '_sequence_manhattan_length')
    df.to_csv(filepath + '.csv')
    
    # bit of a hack as I can't work out how to plot a histogram of grouped data!
    # reformatting for histogram
    hist_list = []
    for i in df.index:
        if df.loc[i].name != 0:
            hist_list = hist_list + [df.loc[i].name] * df.loc[i, "target_sequence_count"]

    fig, ax = plt.subplots()
    sns.set_theme(style="whitegrid")

    bins=np.arange(0.475, 1.575, 0.05) # 1 is the optimal length.
    sns.histplot(hist_list, bins=bins, ax=ax, edgecolor = "black")
    

    ax.set_xlabel('Normalised Trapline Length')
    ax.set_xlim(0.95, 1.6)

    ax.set_title(experiment_name, fontsize=10)

    ax.set_yscale('log')
    ax.set_ylim(0, num_runs_in_experiment)
    ax.set_ylabel('Logarithmic Count of Traplines')

    ax.xaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    ax.grid(visible=True)
    
    fig.suptitle("Trapline Length Histogram")
    fig.savefig(filepath + '.png')

    #plt.subplots_adjust(left=0.1, right=0.9, top=0.83, bottom=0.15)
    plt.pause(0.00000000001)