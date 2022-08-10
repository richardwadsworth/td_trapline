import gym
class AgentReward(gym.Wrapper):
    def __init__(self, env, size, nest_index, target_indices, reward_delay=50, respiration_reward=0, stationary_reward=0, revisit_inactive_target_reward = 0, change_in_orientation_reward=0):
        """

        PARAMS
        ------
        env: OpenAI gym environment
        size: Arena size (width of square arena)
        nest_index: this index of the nest
        target_indices: the indices of the targets within the arena)
        reward_delay: the number of episodes steps to wait until a visited target's reward is reinstated
        respiration_reward: the reward per time step (normally negative)
        stationary_reward: the reward for remaining stationary in a given time step (normally negative)
        revisit_inactive_target_reward: the reward to revisiting an inactive target (normally negative)
        change_in_orientation_reward: a reward for changing direction (normally negative)
        """
        super().__init__(env, new_step_api=True)
        self.size = size
        self.nest_index = nest_index
        self.target_indices = target_indices
        self.target_indices_local = self.target_indices.copy()
        self.reward_delay = reward_delay
        self.respiration_reward = respiration_reward
        self.stationary_reward = stationary_reward
        self.revisit_inactive_target_reward = revisit_inactive_target_reward
        self.change_in_orientation_reward = change_in_orientation_reward
        self.targets_found_order = []
        self.observations = []
        self.goal_rewards = {key: {'step_count':-1} for (key) in target_indices} # set default rewards


    def update_probability_matrix(self, MDP):
        """
        overwrite the default rewards
        """
        for R in MDP['targets']:
            index, reward = R
            for i in range(self.env.observation_space[0].n):
                for j in range(self.env.action_space.n):
            
                    sar = self.env.P[i][j][0]
                    if sar[1][0] == index: # compare position in state variable
                        self.env.P[i][j] = [(sar[0], sar[1], reward, sar[3])]
                        
        

    def step(self, action):
        """
        PARAMS:
        action: action to take
        stats:  whether to record stats about the current episode
        """
        obs, reward, done, truncated, info = self.env.step(action)
        index = obs[0]

        done= False
        if index in self.target_indices_local: # the agent has found a goal
            
            self.target_indices_local.remove(index)

            
            self.goal_rewards[index]['step_count'] = self.env._elapsed_steps #record when this goal was found
            self.targets_found_order.append(index) # record the id of the target
            info["Target.found"] = True # update info to show an active target was found

            done = (len(self.target_indices_local)==0)

        else:
            # not done yet
            reward = 0

        ## other rewards

        # respiration reward.  a negative reward for every time step
        reward = reward + self.respiration_reward

        # movement reward.  a positive reward if agent moves, to encourage exploration
        if self.env.last_state[0] == obs[0]: #agent has not moved (i.e. is stationary in this time step)
            reward = reward + self.stationary_reward
        
        # orientation reward.  a negative reward if orientation changes
        if self.env.last_state[1] != obs[1]: #agent is changed direction
            reward = reward + self.change_in_orientation_reward

        
        self.observations.append(obs)
            
        return obs, reward, done, truncated, info

    def reset(self,*,seed = None):
        val = super().reset(seed=seed)
        self.targets_found_order = []
        self.target_indices_local = self.target_indices.copy()
        self.observations = [val] # prime observations with the starting point
        self.goal_rewards = {key: {'reward':1, 'step_count':-1} for (key) in self.target_indices} # set default rewards
        return val