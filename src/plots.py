import contextlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from enum import Enum
from IPython.display import display, clear_output
from sklearn.preprocessing import Normalizer
from timeColouredPlots import doColourVaryingPlot2d
from utils import map_index_to_coord
from foraging_agent import ActionType
from utils import map_index_to_coord

class PlotType(Enum):
    NoPlots = 0
    Minimal = 1
    Partial = 2
    Full = 3

def initialise_plots(env):

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
    
    
    # set up the subplots for visualisation
    fig1, axs = plt.subplots(2,3, figsize=(11,7))
    
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
    ax1.scatter(nest_x,nest_y, c='g', s=100, marker='^') #origin
    ax2.scatter(nest_x,nest_y, c='g', s=100, marker='^') #origin
    
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
    ax3.scatter(nest_x,nest_y, c='g', s=100, marker='^') #origin
    
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
        x_ = x * np.random.normal(1,sigma,len(x))
        y_ = y * np.random.normal(1,sigma,len(y))
        ax.plot(x_,y_, color="black", alpha=alpha, linewidth=linewidth)

    for observations in data:
        #extract the index data from the observations
        coords = [[xs_coordinate_map[observation[0]],ys_coordinate_map[observation[0]]] for observation in observations]
        x = [coord[0] for coord in coords]
        y = [coord[1] for coord in coords]
        plot(x, y)

    ax.set_xlim(-1,env.size)
    ax.set_ylim(-1,env.size)
    ax.set_title("Traffic plot of Agent under {}".format(title))

    nest_x, nest_y = map_index_to_coord(env.size, env.nest_index)
    ax.scatter(nest_x,nest_y, c='g', s=100, marker='^') #origin
    ax.scatter(xs_target,ys_target, c='r', s=100, marker='o') #goal
    
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
    ax.scatter(nest_x,nest_y, c='g', s=100, marker='^') #origin
    
    ax.scatter(xs_target,ys_target, c='r', s=100, marker='o') #goal
    
    ax.invert_yaxis()

    with contextlib.redirect_stdout(None):
        display(fig)    
    clear_output(wait = True)
    plt.pause(0.0000000001)




    
    
    
    
    
