import contextlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
import seaborn as sns
from json import loads

from enum import Enum
from IPython.display import display, clear_output
from sklearn.preprocessing import Normalizer
from timeColouredPlots import doColourVaryingPlot2d
from trapline import RouteType
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
    ax1.scatter(nest_x,nest_y, c='brown', s=100, marker='^') #origin
    ax2.scatter(nest_x,nest_y, c='brown', s=100, marker='^') #origin
    
    ax1.scatter(xs_target,ys_target, c='r', s=100, marker='o') #goal
    ax2.scatter(xs_target,ys_target, c='r', s=100, marker='o') #goal
    
    QP = ax1.quiver(ii,jj, U_norm, V_norm, scale=10)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    ax1.title.set_text('Normalised and scaled')


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
    ax3.scatter(nest_x,nest_y, c='brown', s=100, marker='^') #origin
    
    ax3.scatter(xs_target,ys_target, c='brown', s=100, marker='o') #goal
    ax3.set_title("Agent path")
    ax3.grid()

    xs, ys = [], []
    for index, orientation in env.observations:
        xs.append(xs_coordinate_map[index])
        ys.append(ys_coordinate_map[index])

    ts = np.arange(0, len(xs))

    doColourVaryingPlot2d(xs, ys, ts, fig1, ax3, map='plasma', showBar=showbar)  # only draw colorbar once
    
    # fix plot axis proportions to equal
    ax3.set_aspect('equal')
    ax3.set_xlim([-1, env.size])
    ax3.set_ylim([0-1, env.size])
    ax3.invert_yaxis()

    with contextlib.redirect_stdout(None):
        display(fig1) 

    clear_output(wait = True)
    plt.pause(0.0000000001)


def plot_performance(fig1, ax, episodes, steps, performance, plot_rate):

    
    ax.plot(plot_rate*np.arange(episodes//plot_rate), performance)
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
    ax.scatter(nest_x,nest_y, c='brown', s=100, marker='^') #origin
    ax.scatter(xs_target,ys_target, c='r', s=100, marker='o') #goal

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
    ax.scatter(nest_x,nest_y, c='brown', s=100, marker='^') #origin
    
    ax.scatter(xs_target,ys_target, c='r', s=100, marker='o') #goal
    
    ax.invert_yaxis()

    with contextlib.redirect_stdout(None):
        display(fig)    
    clear_output(wait = True)
    plt.pause(0.0000000001)

def plot_route( fig, ax, size, nest, targets, route, route_type, subtitle):
    
    remove_axis_ticks(ax)

    route_coords = [map_index_to_coord(size, index) for index in route]
    target_coords = [map_index_to_coord(size, index) for index in targets]
    
    from utils import get_sliding_window_sequence
    sequence = get_sliding_window_sequence(2,len(route_coords))

    ax.set_xlim(0,size)
    ax.set_ylim(0,size)

    nest_x, nest_y = map_index_to_coord(size, nest)    
    ax.scatter(nest_x,nest_y, c='brown', s=100, marker='^') #origin

    xs_target = [coord[0] for coord in target_coords]
    ys_target = [coord[1] for coord in target_coords]
    ax.scatter(xs_target,ys_target, c='r', s=100, marker='o') #goal
    
    if route_type == RouteType.Optimal:
        color = 'g' 
        linestyle = 'solid' 
        label= 'Optimal' 
    
    elif route_type == RouteType.SubOptimal:

        color = 'b' 
        linestyle = 'dashed'
        label= 'Sub Optimal'
    else:
        color = 'grey' 
        linestyle = 'dashed'
        label= 'Incomplete'

    for window in sequence:
        xy1 = route_coords[window[0]]
        xy2 = route_coords[window[1]]
        t = ax.annotate("", xy=xy2, xytext=xy1,
                arrowprops=dict(color=color, arrowstyle="->", lw=2.5, linestyle=linestyle))
    
    ax.plot([0],[0], label=label, color=color, lw=2.5, linestyle=linestyle)
    ax.legend(loc='upper left')
    
    ax.set_title("Route " + subtitle)
    fig.tight_layout()
    ax.invert_yaxis()
    
    with contextlib.redirect_stdout(None):
        display(fig)    
    clear_output(wait = True)
    plt.pause(0.0000000001)


def plot_trapline_distribution(experiment_name, num_runs_in_experiment, MDP, route_count_for_experiment, optimal_trapline, optimal_trapline_reversed):

    LABEL_NO_ROUTE_FOUND = 'Invalid Route'

    # reformat data frame for plotting
    df = pd.DataFrame(route_count_for_experiment)
    df['count'] = df['route']
    df['route'] = [loads(d) for d in df.index.to_list()]
    df.index = np.arange(0, len(df))

    # build x-axis labels
    counter = 1
    x_axis = []
    for i, r in enumerate(df['route']):
        if r == []:
            x_axis.append(LABEL_NO_ROUTE_FOUND)
        else:
            x_axis.append(str(counter))
            counter += 1
    df['x-axis'] = x_axis

    # determine the optimality of each route
    def get_route_type(target_sequence):
        if (target_sequence == optimal_trapline) or (target_sequence == optimal_trapline_reversed):
            return RouteType.Optimal
        elif len(target_sequence) == len(optimal_trapline):
            return RouteType.SubOptimal
        else:
            return RouteType.Incomplete

    df['route_type'] = [get_route_type(route) for route in df['route']]

    #plot bar chart
    fig1, ax = plt.subplots(1,1)
    sns.set_theme(style="whitegrid")

    bar_list = ax.bar(df['x-axis'], df['count']) # plot the bar chart
    
    ax.set_xlabel('Routes')

    ax.set_yscale('log')
    ax.set_ylim(0, num_runs_in_experiment)
    ax.set_ylabel('Logarithmic Count of Routes')



    ax.grid()
    #ax.set_ylabel('Count of Routes')
    fig1.suptitle("Trapline Distribution by Route")
    ax.set_title(experiment_name, fontsize=10)

    # highlight the optimal traplines if present in the results
    for i in range(len(df)):
        label = df['x-axis'][i]
        route = df['route'][i]
        route_type = df['route_type'][i]
        if route_type == RouteType.Incomplete:
            bar_list[i].set_color('grey')
        elif route_type == RouteType.Optimal:
            bar_list[i].set_color('g')
        elif route_type == RouteType.SubOptimal:
            bar_list[i].set_color('b')

    ax.set_xticklabels(df['x-axis'], rotation = 90)
    fig1.tight_layout()
    ax.grid()

    # drop the route count with no discernable target based route found
    df = df.drop(df[df['x-axis'] == LABEL_NO_ROUTE_FOUND].index)

    # if there are more than 9 routes, choose the optimal and 7 random
    MAX_NUM_ROUTE_PLOTS = 9
    if len(df)>MAX_NUM_ROUTE_PLOTS:
        #find index of optimal routes
        df_optimal = df.drop(df[df['route_type'] != RouteType.Optimal].index)
        
        remaining = MAX_NUM_ROUTE_PLOTS - len(df_optimal)

        # get suboptimal routes and select a random selection for display
        df_suboptimal = df.drop(df[df['route_type'] != RouteType.SubOptimal].index)
        suboptimal_indexes_for_plot = np.random.choice(df_suboptimal.index, size=remaining, replace=False)
        df_suboptimal_for_plot= df_suboptimal.loc[suboptimal_indexes_for_plot]
        df = pd.concat([df_optimal, df_suboptimal_for_plot])
        df = df.sort_index()
     
    plot_size = int(np.ceil(np.sqrt(len(df))))
    fig2, axs = plt.subplots(plot_size, plot_size, figsize=(plot_size*3, plot_size*3))
    sns.set_theme(style="whitegrid")
    
    fig2.suptitle("Route Lookup for Trapline Distribution by Route\n" + experiment_name)


    axs = np.array(axs).reshape(-1)

    for i, ax in enumerate(axs):

        if i >= len(df):
            break
        # Convert string representation of route to list using json
        route = df['route'][df.index[i]]
        label= df['x-axis'][df.index[i]]
        route_type= df['route_type'][df.index[i]]

        
        
        plot_route(fig2, ax, MDP["size"],MDP["nest"], optimal_trapline, route, route_type, str(label))

    plt.subplots_adjust(left=0.1, right=0.9, top=0.86, bottom=0.15)
    plt.pause(0.00000000001)

def plot_c_Scores(experiment_name, plot_rate, smoothed):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5))
    sns.set_theme(style="whitegrid")

    for c_score, c_score_prime in smoothed:

        xs = np.arange(0, len(c_score)) * plot_rate
       
        alpha = 0.7
        ax1.plot(xs, c_score, alpha=alpha)
        ax2.plot(xs, c_score_prime, alpha=alpha)

    
    # calculate the mean C score across all runs
    df = pd.DataFrame([x[0] for x in smoothed])
    c_score_mean = df.mean() 
    xs = np.arange(0, len(c_score_mean)) * plot_rate
    alpha = 1
    ax1.plot(xs, c_score_mean, alpha=alpha, lw=2, color='black', label="Mean C score")
    ax1.legend(loc='upper right')

    df = pd.DataFrame([x[1] for x in smoothed])
    c_score_prime_primt_mean = df.mean() 
    xs = np.arange(0, len(c_score_prime_primt_mean)) * plot_rate
    alpha = 1
    ax2.plot(xs, c_score_prime_primt_mean, alpha=alpha, lw=2, color='black', label="Mean C score rate of change")
    ax2.legend(loc='upper right')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Rate of change of C Score')
    ax1.set_title("C Score per episode for all runs", fontsize=10)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('(Smoothed) C Score')
    ax2.set_title("Rate of change of C Score per episode for all runs", fontsize=10)


    fig.suptitle("Route C Score by Episode\n" + experiment_name, fontsize=12)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.83, bottom=0.15)
    plt.pause(0.00000000001)

def plot_c_score_distribution(experiment_name, plot_rate, stability_threshold, c_score_indexes):

    fig, ax = plt.subplots()
    sns.set_theme(style="whitegrid")

    c_score_indexes = np.array(c_score_indexes) * plot_rate # scale up from the plot (sample) rate

    sns.histplot(c_score_indexes, bins=20, ax=ax)
    
    ax.set_xlabel('Episode That C Score Stabilises')
    ax.set_ylabel('Frequency')
    fig.suptitle("C Score Stability Histogram\n(Stability Threshold {})".format(stability_threshold))
    ax.set_title(experiment_name, fontsize=10)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.83, bottom=0.15)
    plt.pause(0.00000000001)
