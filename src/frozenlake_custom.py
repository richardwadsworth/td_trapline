# %%
import numpy as np

import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '/Users/richard/Documents/projects/py_learning/sussex/Dissertation/gym')

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

from plots import plotAgentPath, plotActionStateQuiver
from gym_utils import initialise_gym, getGoalCoordinates

from rl_td import action_softmax, get_action_softmax
from q_function import initialise_q


# %%
# parameters for sarsa(lambda)
episodes = 20000
STEPS = 300
gamma = 0.9
alpha = 0.05
# epsilon_start = 0.2
# epsilon_end = 0.001
# epsilon_annealing_stop = int(episodes/2)
eligibility_decay = 0.3
tau = 1 #softmax temperature


env, goal_indices = initialise_gym(STEPS)

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
# def action_epsilon_greedy(q, s, epsilon=0.05):
#     if np.random.rand() > epsilon:
#         return np.argmax(q[s])
#     return np.random.randint(4)

# def get_action_epsilon_greedy(epsilon):
#     return lambda q,s: action_epsilon_greedy(q, s, epsilon=epsilon)

# %%


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

q = initialise_q(env)

performance = np.ndarray(episodes//STEPS)


    

# %%


fig1, axs = plt.subplots(2,2, figsize=(7,7))
ax1, ax2, ax3, ax4 = axs.ravel()

xs_target, ys_target = [],[]
for index in goal_indices:
    x, y = getGoalCoordinates(index, env.observation_space.n)
    xs_target.append(x)
    ys_target.append(y)



# %%

for episode in range(episodes):

    # inew = min(episode,epsilon_annealing_stop)
    # epsilon = (epsilon_start * (epsilon_annealing_stop - inew) + epsilon_end * inew) / epsilon_annealing_stop
    
    E = np.zeros((env.observation_space.n, env.action_space.n))
    
    state = env.reset()
    # action = action_epsilon_greedy(q, state, epsilon)
    action = action_softmax(q, state, tau)
    

    while True:

        E = eligibility_decay * gamma * E
        E[state, action] += 1
        
        new_state, reward, done, info = env.step(action)
        
        # new_action = action_epsilon_greedy(q, new_state, epsilon)
        new_action = action_softmax(q, new_state, tau)

        delta = reward + gamma * q[new_state, new_action] - q[state, action]
        q = q + alpha * delta * E 

        state, action = new_state, new_action

        # if reward > 0:
        #     plotAgentPath(env, fig1, ax3, ax4, xs_target,ys_target)

        if done:
            break

    # only for plotting the performance, not part of the algorithm 
    if episode%STEPS == 0 or episode == episodes-1:
        # performance[episode//STEPS] = average_performance(get_action_epsilon_greedy(epsilon), q=q)
        performance[episode//STEPS] = average_performance(get_action_softmax(tau), q=q)
    
    if episode%STEPS == 0 or episode == episodes-1:
        fig1.suptitle("Episode {}".format(episode))
        plotAgentPath(env, fig1, ax3, ax4, xs_target,ys_target)
        plotActionStateQuiver(env, q, fig1, ax1, ax2, xs_target,ys_target)





# %%

plotAgentPath(env, fig1, ax3, ax4, xs_target,ys_target)
plotActionStateQuiver(env, q, fig1, ax1, ax2,xs_target,ys_target)
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
    if x in goal_indices:
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


