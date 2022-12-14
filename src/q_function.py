import numpy as np
from enum import Enum

from policies import GreedyFlattenedPolicy
from gym.envs.toy_text.foraging_agent import ActionType

# %%
actionsDict = {}
actionsDict[ActionType.WEST.value] = " w "
actionsDict[ActionType.SOUTH.value] = " s "
actionsDict[ActionType.EAST.value] = " e "
actionsDict[ActionType.NORTH.value] = " n "
actionsDict[ActionType.NONE.value] = " - "

actionsDictInv = {}
actionsDictInv["W"] = ActionType.WEST.value
actionsDictInv["S"] = ActionType.SOUTH.value
actionsDictInv["E"] = ActionType.EAST.value
actionsDictInv["N"] = ActionType.NORTH.value
actionsDictInv["-"] = ActionType.NONE.value


def initialise_actor(env):
    return np.zeros((env.observation_space[1].n, env.observation_space[0].n, env.action_space.n))

def initialise_critic(env, rng):
    return rng.uniform(low=0.0, high=0.0000000009, size=(env.observation_space[1].n, env.observation_space[0].n))

def initialise_q(env):
    q = np.ones((env.observation_space[1].n, env.observation_space[0].n, env.action_space.n))

    # Set q(hole,*) equal to 0
    for k in range(env.observation_space[1].n):
        q_index=0
        for i in env.desc:
            for j in range(len(i)):
                if i[j] ==b"H" :#or i[j]==b"G":
                    q[k][q_index,:] = 0.0
                q_index +=1
        
    return q

def get_q_pretty_print(env, q_mean):

    q_mean = np.round(q_mean,3)
    multiline = ""
    multiline += "(A,S) Value function =" + str(q_mean.shape) +"\n"

    for i, j in enumerate(np.arange(0, env.observation_space[0].n, env.size)):
        multiline += "Row {}".format(i) + "\n"  
        multiline += str(q_mean[j:j+env.size,:]) + "\n" 

    return multiline


def get_optimal_q_policy_pretty_print(env, q_mean):
    
    policy = GreedyFlattenedPolicy(env)

    def resolveActionDict(q, s):
        if s in env.target_indices:
            return actionsDict[policy.action(q, s)].upper() #" ! "
        elif all(a == 0 for a in q[s]):
            return " - "
        else:
            return actionsDict[policy.action(q, s)]

    # get the optimal policy
    policyFound = [resolveActionDict(q_mean, s) for s in range(env.observation_space[0].n)]

    multiline = ""
    multiline += "Greedy policy found:" + "\n"
    idxs = np.arange(0, env.observation_space[0].n, env.size)
    for idx in idxs:
        row = []
        for i in range(env.size):
            row.append(policyFound[idx+i])
        multiline += ','. join(row) + "\n"
            
    return multiline