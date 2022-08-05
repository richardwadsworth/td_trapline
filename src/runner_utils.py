import numpy as np
from gym_utils import initialise_gym
from rl_td import train
from q_function import initialise_actor, initialise_critic, get_q_pretty_print, get_optimal_q_policy_pretty_print
from policies import GreedyDirectionalPolicy, SoftmaxDirectionalPolicy
from plots import PlotType

from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray import tune
import mlflow


def train_fn(config):

    is_stochastic = False
    plot_data = None
    plot_rate = 5 # rate at which to plot predictions
    do_in_episode_plots=PlotType.NoPlots
    do_summary_print = True

    rng = np.random.default_rng() # random number generator

    # each parallel run needs its own env instance
    env_local = initialise_gym(int(config["size"]), 
        config["MDP"],
        is_stochastic, 
        float(config["respiration_reward"]), 
        float(config["stationary_reward"]), 
        float(config["revisit_inactive_target_reward"]), 
        float(config["change_in_orientation_reward"]), 
        int(config["steps"]))
    
    # initialise the action state values
    actor = initialise_actor(env_local)
    critic = initialise_critic(env_local, rng)

    policy_train = SoftmaxDirectionalPolicy(env_local, rng)
    policy_predict = GreedyDirectionalPolicy(env_local)

    sim_data = []
    actor, performance = train(env_local, 
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

    env_local.close()

    return {"score_softmax": softmaxPolicyAvgPerf, 
            "score_softmax_last_05": softmaxPolicyAvgLast05Perf,
            "score_softmax_last_20": softmaxPolicyAvgLas20XPerf,
            "score_greedy": greedyPolicyAvgPerf, 
            "pi_optimal__flattened": pretty_print_optimal_q,
            "pi_flattened": pretty_print_q
            }


def train_parallel(num_samples,
                    size, 
                    MDP, 
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

    analysis = tune.run(
        train_fn,
        num_samples=num_samples,
        mode="max",
        config={
            # define search space here
            "size": size,
            "MDP": MDP,
            "experiment_name": experiment_name,
            "respiration_reward": tune.grid_search(respiration_reward),
            "stationary_reward": tune.grid_search(stationary_reward),
            "revisit_inactive_target_reward": tune.grid_search(revisit_inactive_target_reward),
            "change_in_orientation_reward": tune.grid_search(change_in_orientation_reward),
            "episodes": tune.grid_search(episodes),
            "steps": tune.grid_search(steps),
            "eligibility_decay": tune.quniform(**eligibility_decay),
            "alpha_actor": tune.grid_search(alpha_actor),
            "alpha_critic": tune.grid_search(alpha_critic),
            "gamma": tune.quniform(**gamma),
            "epsilon_start": tune.choice([epsilon_start]),
            "epsilon_end": tune.choice([epsilon_end]),
            "epsilon_annealing_stop_ratio": tune.quniform(**epsilon_annealing_stop_ratio),
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

    analysis = tune.run(
        train_fn,
        num_samples=num_samples,
        mode="max",
        config=config,
        callbacks=[MLflowLoggerCallback(
            experiment_name=experiment_name,
            save_artifact=True)])

    return analysis