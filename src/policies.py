import numpy as np

class Policy(object):

    def __init__(self, env) -> None:
        self.env = env

    @staticmethod
    def average_performance(env, policy_fct, q):
    
        acc_returns = 0.
        n = 500
        for i in range(n):
            done = False
            s = env.reset()
            while not done:
                a = policy_fct(q, s)
                s, reward, done, info = env.step(a)
                acc_returns += reward

        return acc_returns/n
        
class GreedyPolicy(Policy):

    def __init__(self, env):
        super().__init__(env)

    def action(self, q, s):
        return np.argmax(q[s])

    def average_performance(self, q):
        avgPerf = Policy.average_performance(self.env, self.action, q=q)
        return avgPerf


class SoftmaxPolicy(Policy):
    def __init__(self, env, rng):
        super().__init__(env)
        self.rng = rng

    def action(self, q, s, tau=1):

        beta = 1 / tau
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

    def get_action(self, tau):
        return lambda q,s: self.action(q, s, tau=tau)

    def average_performance(self, func, q):
        avgPerf = Policy.average_performance(self.env, func, q=q)
        return avgPerf

    