import numpy as np
from plots import plotAgentPath, plotActionStateQuiver, PlotType


def train(env, 
        episodes, 
        steps, 
        eligibility_decay, 
        alpha_actor, 
        alpha_critic, 
        gamma, 
        epsilon_start, 
        epsilon_end, 
        epsilon_annealing_stop_ratio, 
        actor, 
        critic, 
        policy_train,
        policy_predict,
        plot_rate,
        plot_data, 
        sim_data,
        do_plot=PlotType.NoPlots,
        record_stats=True,
        clip=False):

    abend=False
    performance_counter = 0

    if do_plot == PlotType.Full:
        #unpack plot objects
        fig1, ax1, ax2, ax3, ax4, _, _, xs_coordinate_map, ys_coordinate_map, xs_target, ys_target = plot_data
        
    performance = np.zeros(episodes//plot_rate) # initialise array to track algorithm's performance

    epsilon_annealing_stop = epsilon_annealing_stop_ratio * episodes

    for episode in range(episodes):

        # anneal the Softmax temperature
        inew = min(episode,epsilon_annealing_stop)
        epsilon = (epsilon_start * (epsilon_annealing_stop - inew) \
               + epsilon_end * inew) / epsilon_annealing_stop

        # initialise eligibility traces matrix to zero
        E_critic = np.zeros((env.observation_space[1].n, env.observation_space[0].n, env.action_space.n))
        E_actor = np.zeros((env.observation_space[1].n, env.observation_space[0].n, env.action_space.n))
        
        # reset the environment
        observation = env.reset()

        # get the first action using the annealed softmax policy
        action = policy_train.action(actor, observation, epsilon)
        
        while True:

            # update the eligibility traces.  Assign a weight of 1 to the last visited state
            E_critic = eligibility_decay * gamma * E_critic
            E_actor = eligibility_decay * gamma * E_actor
            
            E_critic[observation[1], observation[0], action] = 1
            E_actor[observation[1], observation[0], action] = 1
            
            # step through the environment
            new_observation, reward, done, truncated, info = env.step(action, record_stats)

            if clip:
                if episode == 75 and performance[performance_counter-1]  < 2.5 and not (done or truncated):
                    print("Abending.. poor performance ({})".format(str(performance[performance_counter-1])))
                    abend = True
                    # reward -= 1
                
            # get the next action using the annealed softmax policy
            new_action = policy_train.action(actor, new_observation, epsilon)

            # Calculate the delta update and update the Q-table using the SARSA TD(lambda) rule:
            td_error = reward + gamma * critic[new_observation[1], new_observation[0], new_action] - critic[observation[1], observation[0], action]
            
            # update the actor and critic q learning tables
            critic = critic + alpha_critic * td_error * E_critic
            actor = actor + alpha_actor * td_error * E_actor 

            # update the state and action values
            observation, action = new_observation, new_action

            if done or truncated or abend:
                break

        # evaluate the agent performance using current actor q learning table (no additional learning)
        if episode%plot_rate == 0 or episode == episodes-1:
            performance[episode//plot_rate] = policy_train.average_performance(policy_train.get_action(epsilon), q=actor)
            performance_counter +=1
            # print("p:{}, e:{}".format(performance[episode//plot_rate], episode))
            
            if record_stats:
                #record stats
                sim_data.append(env.observations)
            
        # evaluate the agent performance and plot
        if episode > 0 and episode%plot_rate == 0 or episode == episodes-1:
            # print("Episode {}. Epsilon {}.".format(episode, epsilon))    
            # shortest_trap_line_count = len([x for x in env.targets_found_order_by_episode if x == env.target_indices]) #check each trap line to see if it is optimal    
            # if shortest_trap_line_count > 0:
            #     print("Total # trap lines: {2}\tOptimal TL:{0}\tOptimal TL as % of all TL ({3}%)\tOptimal TL as % of all episodes ({1}%)".format(\
            #     shortest_trap_line_count, \
            #         np.round(shortest_trap_line_count/episode*100,2), \
            #             len(env.targets_found_order_by_episode), \
            #                 np.round(shortest_trap_line_count/len(env.targets_found_order_by_episode)*100,2)) 
            #         ) 
            
            if do_plot == PlotType.Full:
                fig1.suptitle("Episode {}".format(episode))
                plotAgentPath(env, fig1, ax3, ax4, xs_coordinate_map, ys_coordinate_map, xs_target,ys_target)
                plotActionStateQuiver(env, actor, fig1, ax1, ax2, xs_target,ys_target)
                # print("Training perf: {}".format(performance[episode//plot_rate]))
                # set the spacing between subplots
                # fig1.tight_layout()
        if abend:
            break         

    return actor, performance, done
