from pickle import FALSE
import numpy as np
from plots import plotAgentPath, plotActionStateQuiver
from policies import SoftmaxPolicy

rng = np.random.default_rng(5) # random number generator

def train(env, episodes, steps, eligibility_decay, alpha, gamma, T, q, plot_data, do_plot=False):

    policy = SoftmaxPolicy(env, T, rng)

    #unpack plot objects
    fig1, ax1, ax2, ax3, ax4, xs_target, ys_target = plot_data
        
    performance = np.ndarray(episodes//steps) # initialise array to track algorithm's performance

    for episode in range(episodes):

        E = np.zeros((env.observation_space.n, env.action_space.n))
        
        state = env.reset()

        action = policy.action(q, state)
        
        while True:

            E = eligibility_decay * gamma * E
            E[state, action] += 1
            
            new_state, reward, done, info = env.step(action, True)
            
            if "Target.found" in info:
                #reset Tau to encourage exploration
                #TODO
                pass

            new_action = policy.action(q, new_state)

            delta = reward + gamma * q[new_state, new_action] - q[state, action]
            q = q + alpha * delta * E 

            state, action = new_state, new_action

            # if reward > 0:
            #     plotAgentPath(env, fig1, ax3, ax4, xs_target,ys_target)

            if done:
                break

        # only for plotting the performance, not part of the algorithm 
        if episode%steps == 0 or episode == episodes-1:
            performance[episode//steps] = policy.average_performance(q=q)
        
        
        if episode > 0 and episode%steps == 0 or episode == episodes-1:
            print("Episode {}".format(episode))    
            shortest_trap_line_count = len([x for x in env.targets_found_order_by_episode if x == env.goal_indices]) #check each trap line to see if it is optimal    
            if shortest_trap_line_count > 0:
                print("Total # trap lines: {2}\tOptimal: {0}\tAs % of total episodes ({1}%)\tAs % of total trap lines ({3}%)".format(shortest_trap_line_count, np.round(shortest_trap_line_count/episode*100,2), len(env.targets_found_order_by_episode), np.round(shortest_trap_line_count/len(env.targets_found_order_by_episode)*100,2)))
            
            if do_plot:
                fig1.suptitle("Episode {}".format(episode))
                plotAgentPath(env, fig1, ax3, ax4, xs_target,ys_target)
                plotActionStateQuiver(env, q, fig1, ax1, ax2, xs_target,ys_target)
                # set the spacing between subplots
                # fig1.tight_layout()
                

    plotAgentPath(env, fig1, ax3, ax4, xs_target,ys_target)
    plotActionStateQuiver(env, q, fig1, ax1, ax2,xs_target,ys_target)
    fig1.tight_layout()

    return q, performance, ax4
