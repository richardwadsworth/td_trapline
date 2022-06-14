import numpy as np
import gym

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