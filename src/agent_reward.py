import gym
class AgentReward(gym.Wrapper):
    def __init__(self, env, size, goal_indices, reward_delay=50, respiration_reward=0, movement_reward=0, change_in_orientation_reward=0):
        """

        PARAMS
        ------
        env: OpenAI gym environment
        size: Arena size (width of square arena)
        goal_indices (the indices of the targets within the arena)
        reward_delay: the number of episodes steps to wait until a visited target's reward is reinstated
        respiration_reward: the reward per time step (normally negative)
        movement_reward: the reward for moving in a time step (normally positive)
        change_in_orientation_reward: a reward for changing direction (normally negative)
        """
        super().__init__(env, new_step_api=True)
        self.size = size
        self.goal_indices = goal_indices
        self.reward_delay = reward_delay
        self.respiration_reward = respiration_reward
        self.movement_reward = movement_reward
        self.change_in_orientation_reward = change_in_orientation_reward
        self.observations = []

        self.target_found = False

        self.targets_found_order = [] # used to record which order the targets are found in an episode
        self.targets_found_order_by_episode = [] # used to record the order for each episode
        self.all_targets_found_total_steps = [] # used to record the total number of steps taken to find all targets
        self.goal_rewards = {key: {'step_count':-1} for (key) in goal_indices} # set default rewards


    def update_probability_matrix(self, rewards):
        """
        update the rewards
        """
        for R in rewards:
            index, reward = R
            for i in range(self.env.observation_space[0].n):
                for j in range(self.env.action_space.n):
            
                    sar = self.env.P[i][j][0]
                    if sar[1][0] == index: # compare position in state variable
                        self.env.P[i][j] = [(sar[0], sar[1], reward, sar[3])]
                        
        

    def step(self, action, stats=False):
        """
        PARAMS:
        action: action to take
        stats:  whether to record stats about the current episode
        """
        obs, reward, done, truncated, info = self.env.step(action)
        index = obs[0]
        if index in self.goal_indices: # the agent has found a goal and done will be True
            
            if not truncated: # not timed out

                # reward = self.goal_rewards[index]['reward']
                if self.goal_rewards[index]['step_count']  == -1: 
                    # active target found

                    self.targets_found_order.append(index) # record the id of the target
                    self.goal_rewards[index]['step_count'] = self.env._elapsed_steps #record when this goal was found

                    info["Target.found"] = True # update info to show an active target was found

                    # check to see if all targets have been found.  i.e. if there are not any undiscovered active targets
                    done = True 
                    for target in reversed(self.goal_rewards.items()): # as an optimisation, check the "last" target first
                        if target[1]['step_count'] == -1:
                            done = False
                            break

                    if done and stats:
                        self.targets_found_order_by_episode.append(self.targets_found_order) # record which order the targets were found
                        # self.all_targets_found_total_steps.append(self.env._elapsed_steps)

                        # if self.goal_indices == self.targets_found_order:
                        #     shortest_route = ""
                        # else:
                        #     shortest_route = "!!NOT SHORTEST ROUTE!!"
                        # print("All targets found in average {} steps. {}".format(int(np.mean(self.all_targets_found_total_steps)),shortest_route))

                elif self.env._elapsed_steps - self.goal_rewards[index]['step_count'] > self.reward_delay:
                    self.goal_rewards[index]['step_count'] = -1 #stop tracking the reward
                    reward = 0
                    done = False # NOT done yet

                else:
                    reward = 0
                    done = False # NOT done yet, there are still undiscovered targets
            
        ## other rewards

        # respiration reward.  a negative reward for every time step
        reward = reward + self.respiration_reward

        # movement reward.  a positive reward if agent moves, to encourage exploration
        if self.env.last_state[0] != obs[0]: #agent has moved
            reward = reward + self.movement_reward
        
        # orientation reward.  a negative reward if orientation changes
        if self.env.last_state[1] != obs[1]: #agent is changed direction
            reward = reward + self.change_in_orientation_reward

        
        self.observations.append(obs)
            
        return obs, reward, done, truncated, info

    def reset(self,*,seed = None):
        val = super().reset(seed=seed)
        self.observations = []
        self.targets_found_order = []
        self.goal_rewards = {key: {'reward':1, 'step_count':-1} for (key) in self.goal_indices} # set default rewards
        return val