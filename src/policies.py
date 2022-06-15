import numpy as np

class Policy(object):

    def __init__(self, env) -> None:
        self.env = env

    def action(self, q, s):
        """
        implement this function in the sub class
        """
        raise NotImplementedError()

    def average_performance(self, q):
    
        acc_returns = 0.
        n = 500
        for i in range(n):
            done = False
            s = self.env.reset()
            while not done:
                a = self.action(q, s)
                s, reward, done, info = self.env.step(a)
                acc_returns += reward

        return acc_returns/n
        
class GreedyPolicy(Policy):

    def __init__(self, env):
        super().__init__(env)

    def action(self, q, s):
        return np.argmax(q[s])

class SoftmaxPolicy(Policy):
    def __init__(self, env, tau, rng):
        super().__init__(env)
        self.tau = tau
        self.rng = rng

    def action(self, q, s):

        beta = 1 / self.tau
        probs = np.exp(q[s]*beta) / np.sum(np.exp(q[s]*beta))
        probs =  probs/ np.sum(probs) # Ensure probs is normalised to 1 (to avoid rounding errors)
        randchoice = self.rng.random()
        flag = 1; k = 1
        while flag:

            if randchoice<np.sum(probs[0:k]):
                action = k-1 # adjust for zero based action index
                flag = 0
            
            k = k + 1

        return action

    