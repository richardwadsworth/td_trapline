import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from timeColouredPlots import doColourVaryingPlot2d
from gym_utils import getGoalCoordinates


# %%
def plotActionStateQuiver(env, q, fig1, ax1, ax2, xs_target, ys_target):

    # %%
    def resolveActionState(actionState):

        #L, D, R, U
        vertical = actionState[3] - actionState[1] 
        horizontal = actionState[2] - actionState[0] 

        return horizontal, vertical

    size=int(np.sqrt(env.observation_space.n))    
    
    policyFound = [resolveActionState(q[x,:]) for x in range(env.observation_space.n)]
    
    
    i = np.arange(0,size) #rows
    j = np.arange(0,size) #colums

    ii, jj = np.meshgrid(i,j)#, indexing='ij')

    # print("row indices:\n{}\n".format(ii))
    # print("column indices:\n{}".format(jj))

    U = np.reshape([i[0] for i in policyFound], (size, size))
    V = np.reshape([i[1] for i in policyFound], (size, size))

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
    # ax.clear()
    plt.pause(0.001)

# %%
def plotAgentPath(env, fig1, ax3, ax4, xs_target, ys_target):
    
    #set up plot
    if ax3.get_title() == "":
        ax4.scatter(xs_target,ys_target, c='brown', s=20, marker='o') #goal
        showbar = True
        
        ax4.set_title("All Agent path")
    else:
        showbar=False

    ax3.cla()
    ax3.scatter(xs_target,ys_target, c='brown', s=20, marker='o') #goal
    ax3.set_title("Agent path")

    xs, ys = [], []
    for i in env.observations:
        x, y = getGoalCoordinates(i, env.observation_space.n)
        xs.append(x)
        ys.append(y)

    ts = np.arange(0, len(xs))

    doColourVaryingPlot2d(xs, ys, ts, fig1, ax3, map='plasma', showBar=showbar)  # only draw colorbar once
    
    # fix plot axis proportions to equal
    ax3.set_aspect('equal')
    ax3.set_xlim([-1, int(np.sqrt(env.observation_space.n))+1])
    ax3.set_ylim([0-1, int(np.sqrt(env.observation_space.n))+1])
    ax3.invert_yaxis()

    # doColourVaryingPlot2d(xs, ys, ts, fig1, ax4, map='plasma', showBar=showbar)  # only draw colorbar once
    # ax4.set_aspect('equal')
    # ax4.set_xlim([-1, int(np.sqrt(env.observation_space.n))+1])
    # ax4.set_ylim([0-1, int(np.sqrt(env.observation_space.n))+1])
    # ax4.invert_yaxis()

    display(fig1)    
    clear_output(wait = True)
    # ax.clear()
    plt.pause(0.001)


def plot_performance(episodes, steps, performance):
    plt.plot(steps*np.arange(episodes//steps), performance)
    plt.xlabel("Epochs")
    plt.title("Learning progress for SARSA")
    plt.ylabel("Average reward of an epoch")
    plt.grid()