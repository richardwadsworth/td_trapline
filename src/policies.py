import numpy as np

class Policy(object):

    def __init__(self, env) -> None:
        self.env = env

    def action(self, q, index):
        """
        implement this function in the sub class
        """
        raise NotImplementedError()

    def average_performance(self, policy_fct, q):
    
        acc_returns = 0.
        n = 500
        for i in range(n):
            done = False
            s = self.env.reset()
            while not done:
                a = policy_fct(q, s)
                s, reward, done, info = self.env.step(a)
                acc_returns += reward

        return acc_returns/n
        
class GreedyDirectionalPolicy(Policy):

    def __init__(self, env):
        super().__init__(env)

    def action(self, q, s):
        return np.argmax(q[s[1], s[0]])


class GreedyFlattenedPolicy(Policy):

    def __init__(self, env):
        super().__init__(env)

    def action(self, q, s):
        return np.argmax(q[s])

class SoftmaxDirectionalPolicy(Policy):
    
    def __init__(self, env, rng):
        super().__init__(env)
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
    
    def __init__(self, env, rng):
        super().__init__(env)
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

