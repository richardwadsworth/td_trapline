import numpy as np
from gym_utils import initialise_gym
from rl_td import train
from q_function import initialise_actor, initialise_critic, get_q_pretty_print, get_optimal_q_policy_pretty_print
from policies import GreedyDirectionalPolicy, ReturnToNestPolicy
from plots import plotAgentPath, plotActionStateQuiver, PlotType
from plots import plot_performance, initialise_plots, plot_traffic_noise
import matplotlib.pyplot as plt

from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray import tune
import mlflow

def train_fn(config):
    """
    Run a simulation in the scope of an mlflow experiment and return the results.
    """

    is_stochastic = False
    plot_data = None
    plot_rate = int(config["plot_rate"]) # rate at which to plot predictions
    do_in_episode_plots=PlotType.NoPlots
    do_summary_print = True

    rng = np.random.default_rng() # random number generator

    # each parallel run needs its own env instance
    env_local = initialise_gym(int(config["size"]), 
        config["MRP"],
        is_stochastic, 
        float(config["respiration_reward"]), 
        float(config["stationary_reward"]), 
        float(config["revisit_inactive_target_reward"]), 
        float(config["change_in_orientation_reward"]), 
        int(config["steps"]))
    
    # initialise the action state values
    actor = initialise_actor(env_local)
    critic = initialise_critic(env_local, rng)

    policy_train = ReturnToNestPolicy(env_local, rng, 50, env_local.size, env_local.nest_index)
    policy_predict = GreedyDirectionalPolicy(env_local)

    sim_data = []
    actor, performance, done = train(env_local, 
        int(config["episodes"]),
        int(config["steps"]),
        float(config["eligibility_decay"]),
        float(config["alpha_actor"]),
        float(config["alpha_critic"]),
        float(config["gamma"]),
        float(config["epsilon_start"]),
        float(config["epsilon_end"]),
        float(config["epsilon_annealing_stop_ratio"]),
        actor,
        critic,
        policy_train,
        policy_predict,
        plot_rate,
        plot_data,
        sim_data,
        do_in_episode_plots)


    # print("Training performance stdev: {}".format(np.std(performance)))

    # # get the final performance value of the algorithm using a greedy policy
    greedyPolicyAvgPerf =policy_predict.average_performance(policy_predict.action, q=actor)
    softmaxPolicyAvgPerf = np.mean(performance)
    softmaxPolicyAvgLast05Perf = np.mean(performance[-5])
    softmaxPolicyAvgLas20XPerf = np.mean(performance[-20])

    #get average action state values across all possible actions.  i.e. get a 2d slice of the 3d matrix
    q_mean = np.mean(actor, axis=(0))
    pretty_print_q = get_q_pretty_print(env_local, q_mean) if do_summary_print else "not printed"

    # # print the optimal policy in human readable form
    pretty_print_optimal_q = get_optimal_q_policy_pretty_print(env_local, q_mean) if do_summary_print else "not printed"

    # if done, get number of steps of last 20 episodes  
    softmaxPolicyAvgLas20XSteps = np.mean([len(x) for x in sim_data[-20:]])
    

    env_local.close()

    return {"score_softmax": softmaxPolicyAvgPerf, 
            "score_softmax_last_05": softmaxPolicyAvgLast05Perf,
            "score_softmax_last_20": softmaxPolicyAvgLas20XPerf,
            "steps_softmax_last_20": softmaxPolicyAvgLas20XSteps,
            "score_greedy": greedyPolicyAvgPerf, 
            "performance": performance,
            "pi_optimal__flattened": pretty_print_optimal_q,
            "pi_flattened": pretty_print_q,
            "observations": sim_data,
            "done":done
            }


def train_parallel(num_samples,
                    size, 
                    MRP, 
                    experiment_name, 
                    respiration_reward,
                    stationary_reward,
                    revisit_inactive_target_reward,
                    change_in_orientation_reward,
                    episodes,
                    steps,
                    eligibility_decay,
                    alpha_actor,
                    alpha_critic,
                    gamma,
                    epsilon_start,
                    epsilon_end,
                    epsilon_annealing_stop_ratio):
    """
        Run a hyper-parameters grid search using the MRP provided    

    """

    analysis = tune.run(
        train_fn,
        num_samples=num_samples,
        mode="max",
        config={
            # define search space here
            "size": size,
            "MRP": MRP,
            "experiment_name": experiment_name,
            "respiration_reward": tune.grid_search(respiration_reward),
            "stationary_reward": tune.grid_search(stationary_reward),
            "revisit_inactive_target_reward": tune.grid_search(revisit_inactive_target_reward),
            "change_in_orientation_reward": tune.grid_search(change_in_orientation_reward),
            "episodes": tune.grid_search(episodes),
            "steps": tune.grid_search(steps),
            "eligibility_decay": tune.grid_search(eligibility_decay),
            "alpha_actor": tune.grid_search(alpha_actor),
            "alpha_critic": tune.grid_search(alpha_critic),
            "gamma": tune.grid_search(gamma),
            "epsilon_start": tune.choice([epsilon_start]),
            "epsilon_end": tune.choice([epsilon_end]),
            "epsilon_annealing_stop_ratio": tune.grid_search(epsilon_annealing_stop_ratio),
            "plot_rate" : 5,
            "mlflow": {
                "experiment_name": experiment_name,
                "tracking_uri": mlflow.get_tracking_uri()}
        },
        callbacks=[MLflowLoggerCallback(
            experiment_name=experiment_name,
            save_artifact=True)])

    return analysis

def train_parallel_with_config(num_samples,
                    experiment_name, 
                    config):
    """
        Run one set of hyper parameter config in parallel

    Args:
        num_samples: total number of simulatinos to run
        experiment_name: mlflow experiment name containing the run id of the run that contains the hyper parameter set used
        config: the Ray Tune hyper parameter set 

    Returns:
        analysis results
    """

    analysis = tune.run(
        train_fn,
        num_samples=num_samples,
        mode="max",
        config=config,
        callbacks=[MLflowLoggerCallback(
            experiment_name=experiment_name,
            save_artifact=True)])

    return analysis

def train_fnn(is_stochastic,
            size, 
            MRP, 
            respiration_reward,
            stationary_reward,
            revisit_inactive_target_reward,
            change_in_orientation_reward,
            episodes,
            steps,
            eligibility_decay,
            alpha_actor,
            alpha_critic,
            gamma,
            epsilon_start,
            epsilon_end,
            epsilon_annealing_stop_ratio,
            plot_rate,
            do_plots,
            record_stats,
            rng,
            threshold=5):
    """
    Run a simulation in a loop until the cumulative new reward is greater than the threshold
    """

    plot_data = None
    env = initialise_gym(size, MRP, is_stochastic, respiration_reward, stationary_reward, revisit_inactive_target_reward, change_in_orientation_reward, steps)

    print("Action space = ", env.action_space)
    print("Observation space = ", env.observation_space)

    while True:

        env.reset()
        # env.render()

        # initialise the action state values
        actor = initialise_actor(env)
        critic = initialise_critic(env, rng)

        if do_plots == PlotType.Full:
            plot_data = initialise_plots(env)

        policy_train = ReturnToNestPolicy(env, rng, 50, env.size, env.nest_index)
        
        policy_predict = GreedyDirectionalPolicy(env)

        sim_data = []
        
        # train the algorithm
        actor, performance, done = train(env, 
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
            do_plots,
            record_stats)

        print("Training performance mean: {}".format(np.mean(performance)))
        print("Training performance stdev: {}".format(np.std(performance)))

        LAST_X = 5


        last_x_mean_performance = np.mean(performance[-LAST_X])
        print("Training performance last {} mean: {}".format(LAST_X, last_x_mean_performance))

        # get number of steps of last X episodes  
        last_x_mean_steps = np.mean([len(x) for x in sim_data[-LAST_X:]])
        print("Step count last {} mean: {}".format(LAST_X, last_x_mean_steps))


        if (do_plots==PlotType.Full or \
                do_plots==PlotType.Partial or \
                do_plots == PlotType.Minimal) and \
                    last_x_mean_steps != steps+1 and \
                    last_x_mean_performance > threshold:

            # plot data needed, but not yet initialised
            fig1, ax1, ax2, ax3, ax4, ax5, ax6, xs_coordinate_map, ys_coordinate_map, xs_target, ys_target = initialise_plots(env)

            plotAgentPath(env, fig1, ax3, ax4, xs_coordinate_map, ys_coordinate_map, xs_target,ys_target) # plot the path of the agent's last episode
            plotActionStateQuiver(env, actor, fig1, ax1, ax2, xs_target,ys_target) # plot the quiver graph of the agent's last episode
            plot_performance(fig1, ax4, episodes, steps, performance, plot_rate) # visual the algorithm's performance
            fig1.tight_layout()

        
        if last_x_mean_steps != steps+1 and last_x_mean_performance > threshold:

            #get average action state values across all possible actions.  i.e. get a 2d slice of the 3d matrix
            q_mean = np.mean(actor, axis=(0))

            # these are useful for de bugging
            #print(get_q_pretty_print(env, q_mean)) # print the final action state values
            #print(get_optimal_q_policy_pretty_print(env, q_mean)) # print the optimal policy in human readable form

            softmaxPolicyAvgPerf, obs_data = policy_train.average_performance_with_observations(policy_train.get_action(epsilon_end), q=actor)
            print("Final Softmax policy SARSA performance =", softmaxPolicyAvgPerf) 

            import os   
            from statistics_utils import save_stats         
            artifact_dir = "./artifacts"
            filename = '{}'.format(hash(os.times()))
            stats_filepath = os.path.join(artifact_dir, filename + '.dat')
            if not os.path.exists(artifact_dir):
                os.mkdir(artifact_dir) 
            
            save_stats(stats_filepath, sim_data) # save this simulation's data to disk
            print("Output file is " + stats_filepath)

            if do_plots!=PlotType.NoPlots:
                plot_traffic_noise(env, fig1, ax5, xs_coordinate_map, ys_coordinate_map, xs_target, ys_target, sim_data, "Training")
                plot_traffic_noise(env, fig1, ax6, xs_coordinate_map, ys_coordinate_map, xs_target, ys_target, obs_data, "Test")

            
            

            if do_plots != PlotType.NoPlots:
                fig_filepath = os.path.join(artifact_dir, filename)
                plt.savefig(fig_filepath)
                plt.show()

            
            print("End.")
            break
        
        else:
            print("Continue...")
            
        print()

        env.close()