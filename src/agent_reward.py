import gym

class AgentReward(gym.RewardWrapper):
    def __init__(self, env, size, goal_indices, reward_delay=50, respiration_reward=0):
        super().__init__(env)
        self.size = size
        self.goal_indices = goal_indices
        self.reward_delay = reward_delay
        self.respiration_reward = respiration_reward
        self.observations = []

        self.goal_rewards = {key: {'reward':1, 'step_count':-1} for (key) in goal_indices} # set default rewards
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if obs in self.goal_indices: # the agent has found a goal
            if "TimeLimit.truncated" not in info: # not timed out
                reward = self.goal_rewards[obs]['reward']
                if self.goal_rewards[obs]['step_count']  == -1: # 
                    self.goal_rewards[obs]['reward'] = 0 # set reward to zero
                    self.goal_rewards[obs]['step_count'] = self.env._elapsed_steps #record when this goal was found
                elif self.env._elapsed_steps - self.goal_rewards[obs]['step_count'] > self.reward_delay:
                    self.goal_rewards[obs]['reward'] = 1 # re-establish the reward
                    self.goal_rewards[obs]['step_count'] = -1 #stop tracking the reward
                
                found = True
                for item in self.goal_rewards.items():
                    if item[1]['step_count'] == -1:
                        found = False
                        break
                if found:    
                    done = True 
                else:
                    done = False # override done.  Keep going
            
        reward = reward + self.respiration_reward

        self.observations.append(obs)
            
        
        return obs, reward, done, info

    def reset(self,*,seed = None):
        val = super().reset(seed=seed)
        self.observations = []
        self.goal_rewards = {key: {'reward':1, 'step_count':-1} for (key) in self.goal_indices} # set default rewards
        return val