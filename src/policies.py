import numpy as np
from foraging_agent import ActionType
class Policy(object):

    def __init__(self, env, num_performance_trials) -> None:
        self.env = env
        self.num_performance_trials = num_performance_trials

    def action(self, q, index):
        """
        implement this function in the sub class
        """
        raise NotImplementedError()

    def average_performance(self, policy_fct, q):
    
        acc_returns = 0.
        
        for i in range(self.num_performance_trials):
            done, truncated = False, False
            s = self.env.reset()
            while not done and not truncated:
                a = policy_fct(q, s)
                s, reward, done, truncated, _ = self.env.step(a)
                acc_returns += reward

        return acc_returns/self.num_performance_trials

    def average_performance_with_observations(self, policy_fct, q):
    
        acc_returns = 0.
        observations = []
        
        for i in range(self.num_performance_trials):
            done, truncated = False, False
            s = self.env.reset()
            while not done and not truncated:
                a = policy_fct(q, s)
                s, reward, done, truncated, _ = self.env.step(a)
                acc_returns += reward
            
            observations.append(self.env.observations)

        return acc_returns/self.num_performance_trials, observations
    
        
class GreedyDirectionalPolicy(Policy):

    def __init__(self, env, num_performance_trials=1):
        super().__init__(env, num_performance_trials)

    def action(self, q, s):
        return np.argmax(q[s[1], s[0]])


class GreedyFlattenedPolicy(Policy):

    def __init__(self, env, num_performance_trials=1):
        super().__init__(env, num_performance_trials)

    def action(self, q, s):
        return np.argmax(q[s])

class SoftmaxDirectionalPolicy(Policy):
    
    def __init__(self, env, rng, num_performance_trials=50):
        super().__init__(env, num_performance_trials)
        self.rng = rng


    def action(self, q, s, T):
        probs = np.exp(q[s[1]][s[0]]/T) / np.sum(np.exp(q[s[1]][s[0]]/T))
        probs =  probs/ np.sum(probs) # Ensure probs is normalised to 1 (to avoid rounding errors)
        randchoice = self.rng.random()
        flag = 1; k = 1
        while flag:

            if randchoice<np.sum(probs[0:k]):
                action = k-1 # adjust for zero based action index
                flag = 0
            
            k = k + 1

        return action

    def get_action(self, T):     
        return lambda q,s: self.action(q, s, T=T)


class ReturnToNestPolicy(SoftmaxDirectionalPolicy):
    def __init__(self, env, rng, num_performance_trials, arena_size, nestIndex, nectar_stomach_limit=10):
        super().__init__(env, rng, num_performance_trials)

        self.arena_size = arena_size
        self.nestIndex = nestIndex
        self.nectar_stomach_limit = nectar_stomach_limit
    
    def action(self, q, s, T, homing_beacon_activated=False):
        """_summary_

        Args:
            q (_type_): _description_
            s (_type_): _description_
            T (_type_): _description_
            homing_beacon_activated (bool, optional): _description_. Defaults to False.
            nestIndex (int, optional): _description_. Defaults to -1.

        Returns:
            _type_: _description_
        """

        if not homing_beacon_activated:
            return super().action(q, s, T)
        else:

            
            # get coords of current location
            # get coords of nest
            # choose the action to take home by the shorted route
            from utils import map_index_to_coord
            current_x, current_y = map_index_to_coord(self.arena_size, s[0])
            nest_x, nest_y = map_index_to_coord(self.arena_size, self.nestIndex)

            #e.g current coords = (10, 7), nest coords =  (4, 2)
            # delta_x 10 - 4 = 6, delta_y = 7 - 2 = 5
            # positive y requires a move North, negative x requires a move South
            # negative x requires a move West, negative x requires a move East

            delta_x = current_x - nest_x
            delta_y = current_y - nest_y

            state_values= np.array([0,0,0,0,0])
            # navigate the x vertices first
            if delta_y > 0:
                state_values[ActionType.NORTH.value] = 1
            elif delta_y < 0:
                state_values[ActionType.SOUTH.value] = 1
            else:
                # agent level with nest.  now navigate the y vertices
                if delta_x > 0:
                    state_values[ActionType.WEST.value] = 1
                elif delta_x < 0:
                    state_values[ActionType.EAST.value] = 1
                else: # agent is at the nest!!
                    state_values[ActionType.NONE.value] = 1
            
            return self.softmax(state_values, T)

    def softmax(self, state_values, T):
        probs = np.exp(state_values/T) / np.sum(np.exp(state_values/T))
        probs =  probs/ np.sum(probs) # Ensure probs is normalised to 1 (to avoid rounding errors)
        randchoice = self.rng.random()
        flag = 1; k = 1
        while flag:

            if randchoice<np.sum(probs[0:k]):
                action = k-1 # adjust for zero based action index
                flag = 0
            
            k = k + 1

        return action

    def average_performance(self, policy_fct, q):
    
        acc_returns = 0.
        
        for i in range(self.num_performance_trials):
            done, truncated = False, False
            nectar_stomach_level = 0 
            homing_beacon_activated = False
            s = self.env.reset()
            while not done and not truncated:
                a = policy_fct(q, s, homing_beacon_activated)
                s, reward, done, truncated, info = self.env.step(a)

                if info.get("Target.found", False):
                    nectar_stomach_level = nectar_stomach_level + 1

                # the the agent's crop capacity is full then it's time to head home
                homing_beacon_activated =  (nectar_stomach_level >= self.nectar_stomach_limit)
            
                acc_returns += reward

        return acc_returns/self.num_performance_trials

    def average_performance_with_observations(self, policy_fct, q):
    
        acc_returns = 0.
        observations = []
        
        for i in range(self.num_performance_trials):
            done, truncated = False, False
            nectar_stomach_level = 0 
            homing_beacon_activated = False
            s = self.env.reset()
            while not done and not truncated:
                a = policy_fct(q, s, homing_beacon_activated)
                s, reward, done, truncated, info = self.env.step(a)

                if info.get("Target.found", False):
                    nectar_stomach_level = nectar_stomach_level + 1

                # the the agent's crop capacity is full then it's time to head home
                homing_beacon_activated =  (nectar_stomach_level >= self.nectar_stomach_limit)
            

                acc_returns += reward
            
            observations.append(self.env.observations)

        return acc_returns/self.num_performance_trials, observations

    def get_action(self, T):     
        return lambda q,s,homing_beacon_activated: self.action(q, s, T=T, homing_beacon_activated=homing_beacon_activated)
