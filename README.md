# Using Temporal Difference Learning To Capture Trapline Learning in Foraging Animals

This code base was submitted as part of an Artificial Intelligence and Adaptive Systems MSc Dissertation at Sussex University in 2022.

## Motivation

Foraging for food is a critical selection process for most goal-based animals, and faced with environmental and physiological constraints and costs, an animal’s ability to make the right decision has long-term consequences.  Research shows that pollinating animals often develop routes, or traplines, to visit patchily distributed resources that restore over time in efficient, stable, predictable sequences.

We use Temporal Difference Reinforcement Learning (TD Learning) to capture trapline learning in foraging animals.  Our model does not explicitly model distances, but instead learns a behavioural policy from only reinforcements in its environment.

We developed a machine learning lifecycle to train a stochastic Actor-Critic TD learning model, using Eligibility Traces and a Softmax behavioural policy, on 2 virtual arenas with different spatial arrangements of patchily distributed food resources that restore over time.

This code base forms part of the supplementary artifacts submitted for a dissertation for a Artificial Intelligence and Adaptive Systems MSc, Department of Informatics, University of Sussex, August 2022.

Project supervisors James Bennett and Thomas Nowotny

## How to Install and Run the Project

The project was built and tested on a 2020 Apple Mackbook M1 running MacOS Monterey 12.3.1

### Download the code
- This base uses submodules.  Clone to code using the command  `git clone –recurse-submodules git@github.com:BrainsOnBoard/td_trapline.git`
### Set up the environment

1. Set up a virtual environment  
- Install miniforge using homebrew https://formulae.brew.sh/cask/miniforge

- Create the conda environment  
Execute `conda env create -f environment_droplet.yml -n td_learning`  
Refer to the conda website for more information (https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

- Activate the conda environment   
Execute `conda activate td_learning`


2. Basic environment test  
- Test the model outside of the mlflow environment.  The model will run in a training loop until the cumulative net reward is greater than the threshold variable value (defined in runner_no_mlflow.py).  
Execute `python ./src/runner_no_mlflow.py mrp_10_negative_array_ohashi`

3. Start mlflow  
Execute `cd <to the root of this td_learning folder>`  
Execute `./mlflow ui`  
Refer to (https://www.mlflow.org/docs/latest/quickstart.html#viewing-the-tracking-ui) for more information

- Browse to http://127.0.0.1:5000/# to see the mlflow ui

### Train the agent on one of the arenas

1. Run the hyper-parameter grid search  
- Choose the arena name from the ./src/mrp.py file.  
-- Positive array -> mrp_10_positive_array_ohashi  
-- Negative array -> mrp_10_negative_array_ohashi  
- Use the arena name as the command line parameter.  
Execute `python ./src/runner_tune.py mrp_10_negative_array_ohashi`

- An experiment will be created in mlflow using the arena name e.g. mrp_10_negative_array_ohashi_1661026904536826

![mlfow ui](images/mlflow_ui_screenshot.png?raw=true "mlflow ui")

2. Select the optimal hyper-parameter set

- Copy the the experiment name created in the previous step e.g mrp_10_negative_array_ohashi_1661026904536826  
- Use the experiment name as the command line parameter  
Execute `python ./src/analyse_hyperparameter_sweep.py mrp_10_negative_array_ohashi_1661026904536826`

- Observe the graph generated by the script, and the x and y axis
- Observe the output in the terminal
- Choose the Run ID that yields the 'best' overall "Mean Last 5 Performance Scores" and "Mean Last 5 Performance Scores".  i.e. Select the hyper-parameter set where the routes have converged, yielding high similar values for both axis i.e. top right data points.  
e.g.  Run with Run id 683d637efb34441dbeb15e9ecbdb4bf6

3. Sample the the optimal hyper-parameter set  

- Copy the the run id selected in the previous step e.g 683d637efb34441dbeb15e9ecbdb4bf6
- Use the run id as the command line parameter  
Execute `python ./src/runner_mlflow.py 683d637efb34441dbeb15e9ecbdb4bf6`  
This will use the selected optimal hyper-parameter set to run a simulation in a loop until the cumulative net reward is greater than the threshold set in runner_mlflow.py.  Change the plotting granularity by changing the do_in_episode_plots variable set in runner_mlflow.py.

4. Kick off 1000 runs using the optimal hyper-parameter set selected in the previous step

- Copy the the run id selected in the previous step e.g 683d637efb34441dbeb15e9ecbdb4bf6
- Use the run id as the command line parameter  
Execute `python ./src/runner_analyse.py 683d637efb34441dbeb15e9ecbdb4bf6`

- This will take about 15 mins depending on your hardware.
- a new experiment will be created in mlflow using the run id and the arena name e.g. 683d_mrp_10_negative_array_ohashi_1661026904124893_16610278550079429

5. Run Analysis scripts

- Copy the the experiment name created in the previous step e.g 683d_mrp_10_negative_array_ohashi_1661026904124893_16610278550079429
- Use the experiment name as the command line parameter  
Execute `python ./src/analyse_average_perf_per_run.py 683d_mrp_10_negative_array_ohashi_1661026904124893_16610278550079429`  
Execute `python ./src/analyse_average_steps_per_run.py 683d_mrp_10_negative_array_ohashi_1661026904124893_16610278550079429`  
Execute `python ./src/analyse_traplines.py 683d_mrp_10_negative_array_ohashi_1661026904124893_16610278550079429`

- All graphs and data are save to ./artifacts using the experiment name
- View the results on realtime in the mlflow ui http://127.0.0.1:5000/# 
