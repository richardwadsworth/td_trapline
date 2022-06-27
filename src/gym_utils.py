import numpy as np

def register_gym():

    from os.path import exists
    import os
    import shutil

    source = "sussex/Dissertation/src/foraging_agent.py"
    destination = "sussex/Dissertation/gym/gym/envs/toy_text/foraging_agent.py"
    if exists(destination):
        os.remove(destination)
    shutil.copyfile(source, destination)

    import gym
    

    gym.envs.register(
     id='ForagingAgent-v1',
     entry_point='gym.envs.toy_text:ForagingAgentEnv')

def initialise_gym(size, MDP, respiration_reward, inactive_reward, max_episode_steps=100):

    # size = 19 # size of grid square
    # mid_point = int(np.floor(size /2))
    # left_point = mid_point - int(np.floor(mid_point /2))
    # right_point = mid_point + int(np.floor(mid_point /2))
    
    

    # goal_indices = [4*size+mid_point, 
    #             6*size+left_point, 6*size+right_point,
    #             9*size+left_point, 9*size+right_point,
    #             12*size+left_point, 12*size+right_point,
    #             14*size+mid_point, 
    #             ]

    
    # straightish line
    # goal_indices = [3*size+left_point, 
    #             9*size+mid_point+1, 
    #             15*size+right_point+2,
                 
    #             ]


    # size = 8
    # goal_indices = int(np.floor(size /2))
                    
    # # curved line
    # goal_indices = [3*size+left_point, 
    #             5*size+mid_point+1, 
    #             7*size+right_point,
    #             # 9*size+right_point+1, 
    #             11*size+right_point+1, 
    #             # 13*size+right_point, 
    #             15*size+right_point-1, 
    #             18*size+right_point-4 
    #             ]


    # size=4
    # goal_indices = [15]

    goal_indices = [int(x) for x in MDP[:,0]]

    import gym
    from agent_reward import AgentReward

    ## note the order of the goal indices is important!! it is used to indicate the shortest route
    desc = generate_random_map_extended(goal_indices, size=size, p=1.0)

    env = gym.make('ForagingAgent-v1', is_slippery=False, max_episode_steps=max_episode_steps, desc=desc)

    wrapped_env = AgentReward(env, size, goal_indices, max_episode_steps+1, respiration_reward, inactive_reward)

    wrapped_env.update_probability_matrix(MDP)
    
    return wrapped_env

def get_goal_coordinates(index, observation_space_size):
    side = int(np.sqrt(observation_space_size))
    x= index%side
    y= int(np.floor(index/side))

    return x, y
    
def generate_random_map_extended(goal_indices: list = None, size: int = 8, p: float = 1.0):
    
    from gym.envs.toy_text.foraging_agent import generate_random_map

    def update_cell(desc, x, y , new_value):
        row = desc[y]
        row_as_list = [row[i] for i in range(len(row))]
        row_as_list[x] = new_value 
        desc[y] = "".join(row_as_list)
        return desc


    # generate a random map with start at [0,0] and goal at [-1, -1]
    desc = generate_random_map(size, p)

    if goal_indices != None:

        desc = update_cell(desc, size-1, size-1, "F") # set the default Goal to frozen
        
        #overwrite the default goal position if goal indices given
        for index in goal_indices:
            x, y = get_goal_coordinates(index, np.square(size))
            desc = update_cell(desc, x, y, "G") # set the cell to be a goal
            
    return desc