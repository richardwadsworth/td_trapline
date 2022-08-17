import argparse
from runner_tune import *
from runner_utils import train_fnn
from plots import PlotType
import mlflow
    
def main():

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()

    run = mlflow.get_run(args.run_id)

    config = run.data.params

    threshold = 9.7 # the cumulative reward needed to stop the simulation loop kicked off by this script

    rng = np.random.default_rng() # random number generator

    is_stochastic = False
    plot_rate = 5 # rate at which to plot predictions
    record_stats = True
    do_in_episode_plots=PlotType.Full 

    size = int(loads(config["MRP"])["size"])

    train_fnn(is_stochastic,
            size,
            config["MRP"], 
            float(config["respiration_reward"]), 
            float(config["stationary_reward"]), 
            float(config["revisit_inactive_target_reward"]), 
            float(config["change_in_orientation_reward"]), 
            int(config["episodes"]),
            int(config["steps"]),
            float(config["eligibility_decay"]),
            float(config["alpha_actor"]),
            float(config["alpha_critic"]),
            float(config["gamma"]),
            float(config["epsilon_start"]),
            float(config["epsilon_end"]),
            float(config["epsilon_annealing_stop_ratio"]),
            plot_rate,
            do_in_episode_plots,
            record_stats,
            rng,
            threshold=threshold)

if __name__ == "__main__":
   main()

