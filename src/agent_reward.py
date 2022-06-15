import gym
import numpy as np

class AgentReward(gym.Wrapper):
    def __init__(self, env, size, goal_indices, reward_delay=50, respiration_reward=0):
        super().__init__(env)
        self.size = size
        self.goal_indices = goal_indices
        self.reward_delay = reward_delay
        self.respiration_reward = respiration_reward
        self.observations = []

        self.targets_found_order = [] # used to record which order the targets are found in an episode
        self.targets_found_order_by_episode = [] # used to record the order for each episode
        self.all_targets_found_total_steps = [] # used to record the total number of steps taken to find all targets
        self.goal_rewards = {key: {'step_count':-1} for (key) in goal_indices} # set default rewards


    def update_probability_matrix(self, rewards):
        """
        update the rewards
        """
        for R in rewards:
            state, reward = R
            for i in range(self.env.observation_space.n):
                for j in range(self.env.action_space.n):
            
                    sar = self.env.P[i][j][0]
                    if sar[1] == state:
                        self.env.P[i][j] = [(sar[0], sar[1], reward, sar[3])]
                        
        

    def step(self, action, stats=False):
        """
        PARAMS:
        action: action to take
        stats:  whether to record stats about the current episode
        """
        obs, reward, done, info = self.env.step(action)
        if obs in self.goal_indices: # the agent has found a goal and done will be True
            if "TimeLimit.truncated" not in info: # not timed out
                # reward = self.goal_rewards[obs]['reward']
                if self.goal_rewards[obs]['step_count']  == -1: # active target found
                    self.targets_found_order.append(obs) # record the id of the target
                    self.goal_rewards[obs]['step_count'] = self.env._elapsed_steps #record when this goal was found
                elif self.env._elapsed_steps - self.goal_rewards[obs]['step_count'] > self.reward_delay:
                    self.goal_rewards[obs]['step_count'] = -1 #stop tracking the reward
                else:
                    reward = 0

                
                if any(item for item in self.goal_rewards.items() if item[1]['step_count'] == -1):    
                    done = False # override done.  Keep going
                else:
                    if stats:
                        self.targets_found_order_by_episode.append(self.targets_found_order) # record which order the targets were found
                        self.all_targets_found_total_steps.append(self.env._elapsed_steps)

                    # if self.goal_indices == self.targets_found_order:
                    #     shortest_route = ""
                    # else:
                    #     shortest_route = "!!NOT SHORTEST ROUTE!!"
                    # print("All targets found in average {} steps. {}".format(int(np.mean(self.all_targets_found_total_steps)),shortest_route))

                    done = True # all targets have been found so stop
           
        reward = reward + self.respiration_reward

        self.observations.append(obs)
            
        
        return obs, reward, done, info

    def reset(self,*,seed = None):
        val = super().reset(seed=seed)
        self.observations = []
        self.targets_found_order = []
        self.goal_rewards = {key: {'reward':1, 'step_count':-1} for (key) in self.goal_indices} # set default rewards
        return val