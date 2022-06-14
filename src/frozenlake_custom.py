# %%
import numpy as np
from numpy.random import random, choice
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '/Users/richard/Documents/projects/py_learning/sussex/Dissertation/gym')

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

from timeColouredPlots import doColourVaryingPlot2d

# %%
# parameters for sarsa(lambda)
episodes = 20000
STEPS = 300
gamma = 0.9
alpha = 0.05
epsilon_start = 0.2
epsilon_end = 0.001
epsilon_annealing_stop = int(episodes/2)
eligibility_decay = 0.3

# %%
def getGoalCoordinates(index, observation_space_size):
    side = int(np.sqrt(observation_space_size))
    x= index%side
    y= int(np.floor(index/side))

    return x, y
    
def generate_random_map_extended(goalIndices: list = None, size: int = 8, p: float = 1.0):
    
    def update_cell(desc, x, y , new_value):
        row = desc[y]
        row_as_list = [row[i] for i in range(len(row))]
        row_as_list[x] = new_value 
        desc[y] = "".join(row_as_list)
        return desc


    # generate a random map with start at [0,0] and goal at [-1, -1]
    desc = generate_random_map(size, p)

    if goalIndices != None:

        desc = update_cell(desc, size-1, size-1, "F") # set the default Goal to frozen
        
        #overwrite the default goal position if goal indices given
        for index in goalIndices:
            x, y = getGoalCoordinates(index, np.square(size))
            desc = update_cell(desc, x, y, "G") # set the cell to be a goal
            
    return desc

# %%
class AgentReward(gym.RewardWrapper):
    def __init__(self, env, respiration_reward, goal_indices, reward_delay=50):
        super().__init__(env)
        self.respiration_reward = respiration_reward
        self.goal_indices = goal_indices
        self.reward_delay = reward_delay
        self.observations = []

        self.goal_rewards = {key: {'reward':1, 'step_count':-1} for (key) in goal_indices} # set default rewards
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if obs in self.goal_indices: # the agent has found a goal
            if "TimeLimit.truncated" not in info: # not timed out
                reward = self.goal_rewards[obs]['reward']
                if self.goal_rewards[obs]['step_count']  == -1: # 
                    self.goal_rewards[obs]['reward'] = 0 # set reward to zero
                    self.goal_rewards[obs]['step_count'] = self.env._elapsed_steps #record when this goal was found
                elif self.env._elapsed_steps - self.goal_rewards[obs]['step_count'] > self.reward_delay:
                    self.goal_rewards[obs]['reward'] = 1 # re-establish the reward
                    self.goal_rewards[obs]['step_count'] = -1 #stop tracking the reward
                
                found = True
                for item in self.goal_rewards.items():
                    if item[1]['step_count'] == -1:
                        found = False
                        break
                if found:    
                    done = True # override done.  Keep going
                else:
                    done = False
        

        reward = reward + self.respiration_reward

        self.observations.append(obs)
            
        
        return obs, reward, done, info

    def reset(self,*,seed = None):
        val = super().reset(seed=seed)
        self.observations = []
        self.goal_rewards = {key: {'reward':1, 'step_count':-1} for (key) in self.goal_indices} # set default rewards
        return val

# %%
size = 19 # size of grid square
mid_point = int(np.floor(size /2))
left_point = mid_point - int(np.floor(mid_point /2))
right_point = mid_point + int(np.floor(mid_point /2))
goalIndices = [2*size+mid_point, 
            6*size+left_point, 6*size+right_point,
            10*size+left_point, 10*size+right_point,
            14*size+left_point, 14*size+right_point,
            18*size+mid_point, 
            ]

# size = 19 # size of grid square
# mid_point = int(np.floor(size /2))
# # goalIndices = [50, 99]
# goalIndices = [6*size+mid_point, 
#             14*size+mid_point 
#             ]


# size=19
# goalIndices = [30, 80, 260]

desc = generate_random_map_extended(goalIndices, size=size, p=1.0)

env = gym.make('FrozenLake-v1', is_slippery=False, max_episode_steps=STEPS, desc=desc)

wrapped_env = AgentReward(env, -0.03, goalIndices, STEPS+1)
env = wrapped_env

# %%
print("Action space = ", env.action_space)
print("Observation space = ", env.observation_space)

# %%
actionsDict = {}
actionsDict[0] = " L "
actionsDict[1] = " D "
actionsDict[2] = " R "
actionsDict[3] = " U "

actionsDictInv = {}
actionsDictInv["L"] = 0
actionsDictInv["D"] = 1
actionsDictInv["R"] = 2
actionsDictInv["U"] = 3

# %%
env.reset()
env.render()

# %%
# optimalPolicy = ["R/D"," R "," D "," L ",
#                  " D "," - "," D "," - ",
#                  " R ","R/D"," D "," - ",
#                  " - "," R "," R "," ! ",]
    
# print("Optimal policy:")
# idxs = [0,4,8,12]
# for idx in idxs:
#     print(optimalPolicy[idx+0], optimalPolicy[idx+1], 
#           optimalPolicy[idx+2], optimalPolicy[idx+3])

# %%
def action_epsilon_greedy(q, s, epsilon=0.05):
    if np.random.rand() > epsilon:
        return np.argmax(q[s])
    return np.random.randint(4)

def get_action_epsilon_greedy(epsilon):
    return lambda q,s: action_epsilon_greedy(q, s, epsilon=epsilon)

# %%
def greedy_policy(q, s):
    return np.argmax(q[s])

# %%
def average_performance(policy_fct, q):
    acc_returns = 0.
    n = 500
    for i in range(n):
        done = False
        s = env.reset()
        while not done:
            a = policy_fct(q, s)
            s, reward, done, info = env.step(a)
            acc_returns += reward
    return acc_returns/n

# %%

q = np.ones((env.observation_space.n, env.action_space.n))
# Set q(hole,*) equal to 0
q_index=0
for i in desc:
    for j in range(len(i)):
        if i[j] =="H" :#or i[j]=="G":
            q[q_index,:] = 0.0
        q_index +=1

performance = np.ndarray(episodes//STEPS)

# %%
def resolveActionState(actionState):

    #L, D, R, U
    vertical = actionState[3] - actionState[1] 
    horizontal = actionState[2] - actionState[0] 

    return horizontal, vertical
    

# %%
from IPython.display import display, clear_output

fig1, axs = plt.subplots(2,2, figsize=(7,7))
ax1, ax2, ax3, ax4 = axs.ravel()

xs_target, ys_target = [],[]
for index in goalIndices:
    x, y = getGoalCoordinates(index, env.observation_space.n)
    xs_target.append(x)
    ys_target.append(y)

def plotActionStateQuiver(fig1, ax1, ax2, xs_target, ys_target):

    size=int(np.sqrt(env.observation_space.n))    
    # fig1, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))
    
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
    
def plotAgentPath(fig1, ax3, ax4, xs_target, ys_target):
    
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

# %%

for episode in range(episodes):

    inew = min(episode,epsilon_annealing_stop)
    epsilon = (epsilon_start * (epsilon_annealing_stop - inew) + epsilon_end * inew) / epsilon_annealing_stop
    
    E = np.zeros((env.observation_space.n, env.action_space.n))
    
    state = env.reset()
    action = action_epsilon_greedy(q, state, epsilon)

    while True:

        E = eligibility_decay * gamma * E
        E[state, action] += 1
        
        new_state, reward, done, info = env.step(action)
        
        new_action = action_epsilon_greedy(q, new_state, epsilon)

        delta = reward + gamma * q[new_state, new_action] - q[state, action]
        q = q + alpha * delta * E 

        state, action = new_state, new_action

        # if reward > 0:
        #     plotAgentPath(fig1, ax3, ax4, xs_target,ys_target)

        if done:
            break

    # only for plotting the performance, not part of the algorithm 
    if episode%STEPS == 0 or episode == episodes-1:
        performance[episode//STEPS] = average_performance(get_action_epsilon_greedy(epsilon), q=q)
    
    if episode%STEPS == 0 or episode == episodes-1:
        fig1.suptitle("Episode {}".format(episode))
        plotAgentPath(fig1, ax3, ax4, xs_target,ys_target)
        plotActionStateQuiver(fig1, ax1, ax2, xs_target,ys_target)





# %%

plotAgentPath(fig1, ax3, ax4, xs_target,ys_target)
plotActionStateQuiver(fig1, ax1, ax2,xs_target,ys_target)
plt.show()

# %%
plt.plot(STEPS*np.arange(episodes//STEPS), performance)
plt.xlabel("Epochs")
plt.title("Learning progress for SARSA")
plt.ylabel("Average reward of an epoch")
plt.grid()

# %%
greedyPolicyAvgPerf = average_performance(greedy_policy, q=q)
print("Greedy policy SARSA performance =", greedyPolicyAvgPerf) 

# %%
q = np.round(q,3)
print("(A,S) Value function =", q.shape)

side =int(np.sqrt(env.observation_space.n))
for i, j in enumerate(np.arange(0, env.observation_space.n, side)):
    print("Row {}".format(i))    
    print(q[j:j+side,:])

# %%
def resolveActionDict(x, actionState):
    if x in goalIndices:
        return " ! "
    elif all(v == 0 for v in actionState):
        return " - "
    else:
        return actionsDict[np.argmax(actionState)]
    

# %%

policyFound = [resolveActionDict(x, q[x,:]) for x in range(env.observation_space.n)]

print("Greedy policy found:")
idxs = np.arange(0, env.observation_space.n, int(np.sqrt(env.observation_space.n)))
for idx in idxs:
    row = []
    for i in range(int(np.sqrt(env.observation_space.n))):
        row.append(policyFound[idx+i])
    print(','. join(row))
        
print(" ")

# print("Optimal policy:")
# idxs = [0,4,8,12]
# for idx in idxs:
#     print(optimalPolicy[idx+0], optimalPolicy[idx+1], 
#           optimalPolicy[idx+2], optimalPolicy[idx+3])

# %%
env.close()


