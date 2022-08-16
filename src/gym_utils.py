from json import loads
import site
import sys
from utils import map_index_to_coord

def register_gym(dev_mode=False):

    from os.path import exists
    import os
    import shutil

    source = "src/foraging_agent.py"
    if dev_mode:
        
        root = "./submodules/gym/"    
        sys.path.insert(1, root)
    else:
        root = site.getsitepackages()[0]
    
    destination =os.path.join(root, "gym/envs/toy_text/foraging_agent.py")
    
    if exists(destination):
        os.remove(destination)
    shutil.copyfile(source, destination)

    #update the end init file
    init_filepath = os.path.join(root, "gym/envs/toy_text/__init__.py")
    reg_text = "from gym.envs.toy_text.foraging_agent import ForagingAgentEnv"
    with open(init_filepath, "r+") as file:
        for line in file:
            if reg_text in line:
                break
        else: # not found, we are at the eof
            file.write(reg_text) # append missing data

    
def initialise_gym(size, MRP, is_stochastic, respiration_reward, stationary_reward, revisit_inactive_target_reward, change_in_orientation_reward, max_episode_steps=100):

    MRP = loads(MRP) #convert from string to json
    nest_index = int(MRP['nest'])
    target_indices = [int(x[0]) for x in MRP['targets']]

    import gym

    gym.envs.register(
     id='ForagingAgent-v1',
     entry_point='gym.envs.toy_text:ForagingAgentEnv')

    from foraging_agent_rewards import ForagingAgentRewards 

    ## note the order of the goal indices is important!! it is used to indicate the shortest route
    desc = generate_random_map_extended(nest_index, target_indices, size=size, p=1.0)

    env = gym.make('ForagingAgent-v1', is_slippery=is_stochastic, max_episode_steps=max_episode_steps, desc=desc, new_step_api=True)

    wrapped_env = ForagingAgentRewards(env, size, nest_index, target_indices, max_episode_steps+1, respiration_reward=respiration_reward, stationary_reward=stationary_reward, revisit_inactive_target_reward=revisit_inactive_target_reward, change_in_orientation_reward=change_in_orientation_reward)

    wrapped_env.update_probability_matrix(MRP)
    
    return wrapped_env

    
def generate_random_map_extended(nest_index: int, target_indices: list, size: int = 8, p: float = 1.0):
    
    res = [ ["F"]*size for i in range(size)] # create default map

    x, y = map_index_to_coord(size, nest_index)
    res[y][x] = "S" # set the nest position

    for index in target_indices:
        x, y = map_index_to_coord(size, index)
        res[y][x] = "G"# set the cell to be a target
    
    desc = ["".join(x) for x in res]
        
    return desc