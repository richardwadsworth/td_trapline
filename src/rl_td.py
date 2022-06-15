import numpy as np
import matplotlib.pyplot as plt
from plots import plotAgentPath, plotActionStateQuiver
from gym_utils import getGoalCoordinates
from policies import SoftmaxPolicy

rng = np.random.default_rng(5) # random number generator

def train(env, episodes, steps, eligibility_decay, alpha, gamma, tau, q):

    policy = SoftmaxPolicy(env, rng)

    fig1, axs = plt.subplots(2,2, figsize=(7,7))
    ax1, ax2, ax3, ax4 = axs.ravel()

    xs_target, ys_target = [],[]
    for index in env.goal_indices:
        x, y = getGoalCoordinates(index, env.observation_space.n)
        xs_target.append(x)
        ys_target.append(y)
        
    performance = np.ndarray(episodes//steps) # initialise array to track algorithm's performance

    for episode in range(episodes):
        
        E = np.zeros((env.observation_space.n, env.action_space.n))
        
        state = env.reset()

        action = policy.action(q, state, tau)
        
        while True:

            E = eligibility_decay * gamma * E
            E[state, action] += 1
            
            new_state, reward, done, info = env.step(action)
            
            new_action = policy.action(q, new_state, tau)

            delta = reward + gamma * q[new_state, new_action] - q[state, action]
            q = q + alpha * delta * E 

            state, action = new_state, new_action

            # if reward > 0:
            #     plotAgentPath(env, fig1, ax3, ax4, xs_target,ys_target)

            if done:
                break

        # only for plotting the performance, not part of the algorithm 
        if episode%steps == 0 or episode == episodes-1:
            performance[episode//steps] = policy.average_performance(policy.get_action(tau), q=q)
        
        if episode%steps == 0 or episode == episodes-1:
            fig1.suptitle("Episode {}".format(episode))
            plotAgentPath(env, fig1, ax3, ax4, xs_target,ys_target)
            plotActionStateQuiver(env, q, fig1, ax1, ax2, xs_target,ys_target)

    plotAgentPath(env, fig1, ax3, ax4, xs_target,ys_target)
    plotActionStateQuiver(env, q, fig1, ax1, ax2,xs_target,ys_target)

    return q, performance
