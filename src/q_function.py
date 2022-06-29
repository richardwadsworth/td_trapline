import numpy as np
from enum import Enum

from policies import GreedyFlattenedPolicy
from gym.envs.toy_text.foraging_agent import ActionType

# %%
actionsDict = {}
actionsDict[ActionType.WEST.value] = " W "
actionsDict[ActionType.SOUTH.value] = " S "
actionsDict[ActionType.EAST.value] = " E "
actionsDict[ActionType.NORTH.value] = " N "
actionsDict[ActionType.NONE.value] = " - "

actionsDictInv = {}
actionsDictInv["W"] = ActionType.WEST.value
actionsDictInv["S"] = ActionType.SOUTH.value
actionsDictInv["E"] = ActionType.EAST.value
actionsDictInv["N"] = ActionType.NORTH.value
actionsDictInv["-"] = ActionType.NONE.value


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

def print_q(env, q_mean):

    q_mean = np.round(q_mean,3)
    print("(A,S) Value function =", q_mean.shape)

    for i, j in enumerate(np.arange(0, env.observation_space[0].n, env.size)):
        print("Row {}".format(i))    
        print(q_mean[j:j+env.size,:])


def print_optimal_q_policy(env, q_mean):
    
    policy = GreedyFlattenedPolicy(env)

    def resolveActionDict(q, s):
        if s in env.goal_indices:
            return " ! "
        elif all(a == 0 for a in q[s]):
            return " - "
        else:
            return actionsDict[policy.action(q, s)]

    # get the optimal policy
    policyFound = [resolveActionDict(q_mean, s) for s in range(env.observation_space[0].n)]

    print("Greedy policy found:")
    idxs = np.arange(0, env.observation_space[0].n, int(np.sqrt(env.observation_space[0].n)))
    for idx in idxs:
        row = []
        for i in range(int(np.sqrt(env.observation_space[0].n))):
            row.append(policyFound[idx+i])
        print(','. join(row))
            
    print(" ")