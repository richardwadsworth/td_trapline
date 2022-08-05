import numpy as np
from gym_utils import register_gym, initialise_gym

register_gym()

from rl_td import train
from q_function import initialise_actor, initialise_critic, get_q_pretty_print, get_optimal_q_policy_pretty_print
from policies import GreedyDirectionalPolicy, SoftmaxDirectionalPolicy
from plots import PlotType
from mdp import *

from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.tune.integration.mlflow import mlflow_mixin
from ray import tune
import mlflow

# parameters for sarsa(lambda)


size, MDP, experiment_name = get_medium_positive_array()
experiment_name = "test"

is_stochastic = False
plot_rate = 5 # rate at which to plot predictions
episodes = [100]
steps = [100, 200, 300]

gamma = {"lower":0.6, "upper":0.9, "q":0.1} # discount factor
alpha_actor = alpha_critic = [0.3, 0.5, 0.7] # actor learning rate, critic learning rate
eligibility_decay = {"lower":0.4, "upper":0.8, "q":0.2}# eligibility trace decay

#softmax temperature annealing
epsilon_start = 1
epsilon_end = 0.2
epsilon_annealing_stop_ratio = {"lower":0.2, "upper":0.8, "q":0.2}

respiration_reward = [-0.01, -0.005] # -1/np.square(size) # -1/(steps+(steps*0.1)) # negative reward for moving 1 step in an episode
stationary_reward = [-0.01, -0.005] # respiration_reward*2 # positive reward for moving, to discourage not moving
revisit_inactive_target_reward = [-0.1, -0.01, 0.0] # negative reward for revisiting an inactive target (i.e. one that has already been visited)
change_in_orientation_reward = [0]#-stationary_reward*0.5 #negative reward if orientation changes

do_in_episode_plots=PlotType.NoPlots
do_summary_print = True
plot_data = None

def train_x(config):

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
    mean_pref = np.mean(performance)  

    #get average action state values across all possible actions.  i.e. get a 2d slice of the 3d matrix
    q_mean = np.mean(actor, axis=(0))
    pretty_print_q = get_q_pretty_print(env_local, q_mean) if do_summary_print else "not printed"

    # # print the optimal policy in human readable form
    pretty_print_optimal_q = get_optimal_q_policy_pretty_print(env_local, q_mean) if do_summary_print else "not printed"

    env_local.close()

    return {"score_softmax": mean_pref, 
            "score_greedy": greedyPolicyAvgPerf, 
            "pi_optimal__flattened": pretty_print_optimal_q,
            "pi_flattened": pretty_print_q
            }
    
    
if __name__ == "__main__":

    analysis = tune.run(
        train_x,
        mode="max",
        num_samples=1,
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

