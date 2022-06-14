import numpy as np

rng = np.random.default_rng(5) # random number generator

def action_softmax(q, s, tau=1):

    # Softmax temperature
    
    beta = 1 / tau
    probs = np.exp(q[s]*beta) / np.sum(np.exp(q[s]*beta))
    probs =  probs/ np.sum(probs) # Ensure probs is normalised to 1 (to avoid rounding errors)
    randchoice = rng.random()
    flag = 1; k = 1
    while flag:

        if randchoice<np.sum(probs[0:k]):
            action = k-1 # adjust for zero based action index
            flag = 0
        
        k = k + 1
        
    return action
    

def get_action_softmax(tau):
    return lambda q,s: action_softmax(q, s, tau=tau)