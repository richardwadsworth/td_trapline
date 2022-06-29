from pickle import FALSE
import numpy as np
from plots import plotAgentPath, plotActionStateQuiver
from policies import SoftmaxDirectionalPolicy

rng = np.random.default_rng(5) # random number generator


def train(env, episodes, steps, eligibility_decay, alpha, gamma, epsilon_start, epsilon_end, epsilon_annealing_stop, q, plot_data, do_plot=False):

    policy = SoftmaxDirectionalPolicy(env, rng)

    #unpack plot objects
    fig1, ax1, ax2, ax3, ax4, xs_target, ys_target = plot_data
        
    performance = np.ndarray(episodes//steps) # initialise array to track algorithm's performance

    for episode in range(episodes):

        # anneal the Softmax temperature
        inew = min(episode,epsilon_annealing_stop)
        epsilon = (epsilon_start * (epsilon_annealing_stop - inew) \
               + epsilon_end * inew) / epsilon_annealing_stop


        # initialise eligibility traces matrix to zero
        E = np.zeros((env.observation_space[1].n, env.observation_space[0].n, env.action_space.n))
        
        # reset the environment
        state = env.reset()

        # get the first action using the annealed softmax policy
        action = policy.action(q, state, epsilon)
        
        while True:

            # update the eligibility traces.  Assign a weight of 1 to the last visited state
            E = eligibility_decay * gamma * E
            E[state[1], state[0], action] += 1
            
            # step through the environment
            new_state, reward, done, info = env.step(action, True)
            
            # get the next action using the annealed softmax policy
            new_action = policy.action(q, new_state, epsilon)

            # Calculate the delta update and update the Q-table using the SARSA TD(lambda) rule:
            delta = reward + gamma * q[new_state[1], new_state[0], new_action] - q[state[1], state[0], action]
            q = q + alpha * delta * E 

            # update the state and action values
            state, action = new_state, new_action

            # if reward > 0:
            #     plotAgentPath(env, fig1, ax3, ax4, xs_target,ys_target)

            if done:
                break

        # evaluate the agent performance
        if episode%steps == 0 or episode == episodes-1:
            performance[episode//steps] = policy.average_performance(policy.get_action(epsilon), q=q)

        # evaluate the agent performance and plot
        if episode > 0 and episode%steps == 0 or episode == episodes-1:
            print("Episode {}. Epsilon {}.".format(episode, epsilon))    
            shortest_trap_line_count = len([x for x in env.targets_found_order_by_episode if x == env.goal_indices]) #check each trap line to see if it is optimal    
            if shortest_trap_line_count > 0:
                print("Total # trap lines: {2}\tOptimal: {0}\tAs % of total episodes ({1}%)\tAs % of total trap lines ({3}%)".format(shortest_trap_line_count, np.round(shortest_trap_line_count/episode*100,2), len(env.targets_found_order_by_episode), np.round(shortest_trap_line_count/len(env.targets_found_order_by_episode)*100,2)))
            
            if do_plot:
                fig1.suptitle("Episode {}".format(episode))
                plotAgentPath(env, fig1, ax3, ax4, xs_target,ys_target)
                plotActionStateQuiver(env, q, fig1, ax1, ax2, xs_target,ys_target)
                # set the spacing between subplots
                # fig1.tight_layout()
                

    
    plotAgentPath(env, fig1, ax3, ax4, xs_target,ys_target) # plot the path of the agent's last episode
    plotActionStateQuiver(env, q, fig1, ax1, ax2,xs_target,ys_target) # plot the quiver graph of the agent's last episode
    fig1.tight_layout()

    return q, performance, ax4
