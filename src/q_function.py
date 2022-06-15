import numpy as np

from policies import GreedyPolicy

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

def initialise_q(env):
    q = np.ones((env.observation_space.n, env.action_space.n))

    # Set q(hole,*) equal to 0
    q_index=0
    for i in env.desc:
        for j in range(len(i)):
            if i[j] ==b"H" :#or i[j]==b"G":
                q[q_index,:] = 0.0
            q_index +=1
    
    return q

def print_q(env, q):
    q = np.round(q,3)
    print("(A,S) Value function =", q.shape)

    for i, j in enumerate(np.arange(0, env.observation_space.n, env.size)):
        print("Row {}".format(i))    
        print(q[j:j+env.size,:])


def print_optimal_q_policy(env, q):
    
    policy = GreedyPolicy(env)

    def resolveActionDict(q, s):
        if s in env.goal_indices:
            return " ! "
        elif all(a == 0 for a in q[s]):
            return " - "
        else:
            return actionsDict[policy.action(q, s)]
            
    # get the optimal policy
    policyFound = [resolveActionDict(q, s) for s in range(env.observation_space.n)]

    print("Greedy policy found:")
    idxs = np.arange(0, env.observation_space.n, int(np.sqrt(env.observation_space.n)))
    for idx in idxs:
        row = []
        for i in range(int(np.sqrt(env.observation_space.n))):
            row.append(policyFound[idx+i])
        print(','. join(row))
            
    print(" ")