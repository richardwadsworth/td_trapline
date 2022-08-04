import numpy as np

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


class SoftmaxFlattenedPolicy(Policy):
    
    def __init__(self, env, rng, num_performance_trials=50):
        super().__init__(env, num_performance_trials)
        self.rng = rng


    def action(self, q, observation, T):

        #get average action state values across all possible actions.  i.e. get a 2d slice of the 3d matrix
        q_mean = np.mean(q, axis=(0))

        probs = np.exp(q_mean[observation[0]]/T) / np.sum(np.exp(q_mean[observation[0]]/T))
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

