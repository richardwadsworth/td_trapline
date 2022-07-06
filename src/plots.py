import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from timeColouredPlots import doColourVaryingPlot2d
from gym_utils import get_goal_coordinates
from gym.envs.toy_text.foraging_agent import ActionType


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
    fig1, axs = plt.subplots(2,2, figsize=(7,7))
    ax1, ax2, ax3, ax4 = axs.ravel()
    remove_axis_ticks(ax1)
    remove_axis_ticks(ax2)
    remove_axis_ticks(ax3)

    # create coordinate lookup table
    xs_coordinate_map, ys_coordinate_map = [], []
    for index in range(env.observation_space[0].n):
        x, y = get_goal_coordinates(index, env.observation_space[0].n)
        xs_coordinate_map.append(x)
        ys_coordinate_map.append(y)

    xs_target, ys_target = [],[]
    for index in env.goal_indices:
        xs_target.append(xs_coordinate_map[index])
        ys_target.append(ys_coordinate_map[index])
    
    return (fig1, ax1, ax2, ax3, ax4, xs_coordinate_map, ys_coordinate_map, xs_target, ys_target)


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

    ax1.scatter([0],[0], c='g', s=100, marker='^') #origin
    ax2.scatter([0],[0], c='g', s=100, marker='^') #origin

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
    ax3.scatter([0],[0], c='g', s=100, marker='^') #origin
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

    display(fig1)    
    clear_output(wait = True)
    plt.pause(0.0000000001)


def plot_performance(episodes, steps, performance, plot_data):

    #unpack plot objects
    fig1, _, _, _, ax4, _, _, _, _= plot_data
    

    ax4.plot(steps*np.arange(episodes//steps), performance)
    ax4.set_xlabel("Epochs")
    ax4.set_title("Learning progress for SARSA")
    ax4.set_ylabel("Average reward of an epoch")
    ax4.grid()
    fig1.tight_layout()
    display(fig1)    
    clear_output(wait = True)
    plt.pause(0.0000000001)
