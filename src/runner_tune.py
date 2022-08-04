import numpy as np
from gym_utils import register_gym, initialise_gym

register_gym()

from rl_td import train
from q_function import initialise_actor, initialise_critic, get_q_pretty_print, get_optimal_q_policy_pretty_print
from policies import GreedyDirectionalPolicy, SoftmaxDirectionalPolicy
from plots import PlotType
from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.tune.integration.mlflow import mlflow_mixin
from ray import tune
import mlflow

# parameters for sarsa(lambda)

def map_coord_to_index(size, x, y):
    return (x*size)+y

# # large straight-ish line
# size = 11
# MDP = np.array([(map_coord_to_index(size, 1, 1),1.0), 
#                 (map_coord_to_index(size, 3, 3),1.0),
#                 (map_coord_to_index(size, 5, 5),1.0),
#                 (map_coord_to_index(size, 7, 7),1.0),
#                 (map_coord_to_index(size, 9, 9),1.0)
#                 ])


# small positive array
size = 7
MDP = np.array([(map_coord_to_index(size, 1, 1),1.0), 
                (map_coord_to_index(size, 3, 1),1.0),
                (map_coord_to_index(size, 1, 3),1.0),
                (map_coord_to_index(size, 5, 3),1.0),
                (map_coord_to_index(size, 3, 5),1.0),
                (map_coord_to_index(size, 5, 5),1.0)
                ])

is_stochastic = False
plot_rate = 5 # rate at which to plot predictions



experiment_name = "james_demo"
episodes = [50, 100]
steps = [100, 150, 200]


gamma = {"lower":0.6, "upper":0.9, "q":0.05} # discount factor
alpha_actor = alpha_critic = [0.3, 0.5, 0.7] # actor learning rate, critic learning rate
eligibility_decay = {"lower":0.1, "upper":0.9, "q":0.05}# eligibility trace decay

#softmax temperature annealing
epsilon_start = 1
epsilon_end = 0.2
epsilon_annealing_stop_ratio = {"lower":0.7, "upper":0.9, "q":0.1}

respiration_reward = -0.01 # -1/np.square(size) # -1/(steps+(steps*0.1)) # negative reward for moving 1 step in an episode
stationary_reward = -0.01 # respiration_reward*2 # positive reward for moving, to discourage not moving
revisit_inactive_target_reward = -0.1 # negative reward for revisiting an inactive target (i.e. one that has already been visited)
change_in_orientation_reward = 0#-stationary_reward*0.5 #negative reward if orientation changes

do_in_episode_plots=PlotType.NoPlots
do_summary_print = True
plot_data = None

def train_x(config):

    rng = np.random.default_rng() # random number generator

    # each parallel run needs its own env instance
    env_local = initialise_gym(size, 
        MDP, 
        is_stochastic, 
        respiration_reward, 
        stationary_reward, 
        revisit_inactive_target_reward, 
        change_in_orientation_reward, 
        config["steps"])
    
    # initialise the action state values
    actor = initialise_actor(env_local)
    critic = initialise_critic(env_local, rng)

    policy_train = SoftmaxDirectionalPolicy(env_local, rng)
    policy_predict = GreedyDirectionalPolicy(env_local)

    sim_data = []
    actor, performance = train(env_local, 
        config["episodes"],
        config["steps"],
        config["eligibility_decay"],
        config["alpha_actor"],
        config["alpha_critic"],
        config["gamma"],
        config["epsilon_start"],
        config["epsilon_end"],
        config["epsilon_annealing_stop_ratio"],
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

    yield {"score_softmax": mean_pref, 
            "score_greedy": greedyPolicyAvgPerf, 
            "pi_optimal__flattened": pretty_print_optimal_q,
            "pi_flattened": pretty_print_q
            }
    
    
# Create the MlFlow expriment.
# mlflow.create_experiment("experiment1")

from ray.tune.suggest.bayesopt import BayesOptSearch

analysis = tune.run(
    train_x,
    mode="max",
    num_samples=5,
    config={
        # define search space here
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

1==1