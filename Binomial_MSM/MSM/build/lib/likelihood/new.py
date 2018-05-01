# -*- coding: utf-8 -*-
#import pdb
import numpy as np
from Binomial_MSM.MSM.likelihood.sharedfunc import gofm
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
#from multiprocessing import Pool
from functools import partial
#from concurrent.futures import ThreadPoolExecutor

#import pdb
class memoize(dict):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result

@memoize
def transition_prob(inpt,state_t,kbar):
    """
    Couputes the transiton probability to all 2^kbar states when supplied with the current state
    
    inpt = all parameters for MSM (b,m0,gamma_kbar,sigma in this order)
    state_t = current state, must be 0 to 2^kbar-1
    kbar = number of multipliers in the model
    """
    b = inpt[0]
    gamma_kbar = inpt[2]
    gamma = np.zeros((kbar,1))
    gamma[0,0] = 1-(1-gamma_kbar)**(1/(b**(kbar-1)))
    for i in range(1,kbar):
        gamma[i,0] = 1-(1-gamma[0,0])**(b**(i))
    gamma = gamma*0.5
    gamma = np.c_[gamma,gamma]
    gamma[:,0] = 1 - gamma[:,0]
    kbar2 = 2**kbar
    prob = np.ones(kbar2)
    # all combination of probabilities from gamma_1*gamma_2*gamma_3, gamma_1*gamma_2*(1-gamma_3),...
    for i in range(kbar2):
        for m in range(kbar):
            tmp = np.unpackbits(np.arange(i,i+1,dtype = np.uint16).view(np.uint8))
            tmp = np.append(tmp[8:],tmp[:8])
            prob[i] =prob[i] * gamma[kbar-m-1,tmp[-(m+1)]]
    A = np.fromfunction(lambda i,j: prob[np.bitwise_xor(np.uint16(state_t),j)],
                          (1,kbar2),dtype = np.uint16)                 
    return(A)
    
def transition_mat_new(inpt,kbar):
    """
    A function that computes the transition matrix for given input and kbar
    
    inpt = all parameters for MSM (b,m0,gamma_kbar,sigma in this order)
    kbar = number of multipliers in the model
    """
    # extract all necessary inputs from inpt and give name
    b = inpt[0]
    gamma_kbar = inpt[2]
    gamma = np.zeros((kbar,1))
    
    # compute gammas
    gamma[0,0] = 1-(1-gamma_kbar)**(1/(b**(kbar-1)))
    for i in range(1,kbar):
        gamma[i,0] = 1-(1-gamma[0,0])**(b**(i))
    gamma = gamma*0.5
    gamma = np.c_[gamma,gamma]
    gamma[:,0] = 1 - gamma[:,0]
    kbar2 = 2**kbar
    prob = np.ones(kbar2)
    
    #generate all combination of probabilities
    for i in range(kbar2):
        for m in range(kbar):
            tmp = np.unpackbits(np.arange(i,i+1,dtype = np.uint16).view(np.uint8))
            tmp = np.append(tmp[8:],tmp[:8])
            prob[i] =prob[i] * gamma[kbar-m-1,tmp[-(m+1)]]
    A = np.fromfunction(lambda i,j: prob[np.bitwise_xor(i,j)],
                          (kbar2,kbar2),dtype = np.uint16)                 
    return(A)

def likelihood_new(inpt,kbar,data,estim_flag,nargout =1):
    """
    Computes the exact likelihood up to the end of the data.
    Depending on the number of inputs it will either return sum of daily log likelihood
    or that and a vector of daily log likelihood.
    The former will be used in starting value calculation while the latter is used in
    parameter estimation and inference.

    inpt = all parameters for MSM (b,m0,gamma_kbar,sigma in this order)
    kbar = number of multipliers in the model
    data = data to use for likelihood calculation
    estim_flag = will be used in starting value calculation, otherwise set it to None
    nargout = number of outputs, default is 1, for other values 3 outputs will be returned
    """

    if not hasattr(inpt,"__len__"):
        inpt = [estim_flag[0],inpt,estim_flag[1],estim_flag[2]]
        
    sigma = inpt[3]/np.sqrt(252)
    k2 = 2**kbar
    A = transition_mat_new(inpt,kbar)
    g_m = gofm(inpt,kbar)
    T = len(data)
    pi_mat = np.zeros((T+1,k2))
    pi_forward = np.zeros(T)
    LLs = np.zeros(T)
    pi_mat[0,:] = (1/k2)*np.ones(k2)
    pa = (2*np.pi)**(-0.5)
    s = sigma*g_m
    w_t = data
    w_t = pa*np.exp(-0.5*((w_t/s)**2))/s
    w_t = w_t + 1e-16
    for t in range(T):
        piA = np.dot(pi_mat[t,:],A)
        pi_forward[t] = np.average(g_m,weights = piA)
        C = (w_t[t,:]*piA)
        ft = np.sum(C)
        if np.isclose(ft,0):
            pi_mat[t+1,1] = 1
        else:
            pi_mat[t+1,:] = C/ft
        
        LLs[t] = np.log(np.dot(w_t[t,:],piA))
    LL = -np.sum(LLs)
    
    if np.any(np.isinf(LLs)):
        print("Log-likelihood is inf. Probably due to all zeros in pi_mat.")
    if nargout == 1:
        return(LL)
    else:
        return(LL,LLs,pi_mat[-1,:],pi_forward)

def sim_one_step(M,inpt,kbar,Ms):
    """
    A function used inside likelihood_pf. It will siumulate next states given current state.
    
    M = an array of current states
    inpt = all parameters for MSM (b,m0,gamma_kbar,sigma in this order)
    kbar = number of multipliers in the model
    Ms = an array of all possible states from 0 to 2^kbar
    """
    
    next_state = []
    for i,v in enumerate(M):
        probs = transition_prob(tuple(inpt),v,kbar)[0]
        next_state.append(np.random.choice(Ms,size = 1, p = probs))
    return(next_state)

def likelihood_pf(inpt,kbar,data,B):
    """
    Computes the simulated likelihood up to the end of the data with particle filter.

    inpt = all parameters for MSM (b,m0,gamma_kbar,sigma in this order)
    kbar = number of multipliers in the model
    data = data to use for likelihood calculation
    B = number of particles to take
    """

    g_m = gofm(inpt,kbar)
    Ms = np.arange(len(g_m))
    sigma = inpt[3]/np.sqrt(252)
    # no particles given so will simulate from time 1to t with data
    k2 = 2**kbar
    T = len(data)
    M_mat = np.zeros((T,B))
    pa = (2*np.pi)**(-0.5)
    s = sigma*g_m
    w_t = data 
    w_t = pa*np.exp(-0.5*((w_t/s)**2))/s
    w_t = w_t + 1e-16
    LLs = np.zeros(T)
    preds = np.zeros(T)
    M_mat[0,:] = np.random.choice(Ms, size=B, replace=True, p=(1/k2)*np.ones(k2))
    LLs[0] = np.mean(w_t[0,M_mat[0,:].astype(int)])
    preds[0] = np.mean(g_m[M_mat[0,:].astype(int)])
    cpu_count = multiprocessing.cpu_count()
    #pdb.set_trace()
    #len_part = B/cpu_count
    pool = ThreadPool(cpu_count)
    for i in range(T-1):
        M_temp = np.zeros(B)
        ws = np.zeros(B)
        #M_parts= [M_mat[i,int(j*len_part):int((j+1)*len_part)] for j in range(4)] 
        #pdb.set_trace()
        #M_temp = np.concatenate(pool.map(partial(sim_one_step,inpt = inpt,
        # kbar = kbar,Ms = Ms),M_parts)).ravel()
        M_temp = np.concatenate(pool.map(partial(sim_one_step,inpt = inpt,kbar = kbar,Ms = Ms),M_mat[i,:].reshape(1,-1))).ravel()
        #for j,val in enumerate(M_mat[i,:]):
        #    probs = transition_prob(tuple(inpt),val,kbar)[0]
        #    M_temp[j] = np.random.choice(Ms,size = 1,p = probs)
        preds[i+1] =np.average(g_m[M_temp.astype(int)],weights = w_t[i+1,M_temp.astype(int)]/np.sum(w_t[i+1,M_temp.astype(int)]))
        LLs[i+1] = np.mean(w_t[i+1,M_temp.astype(int)])
        for k,val in enumerate(M_temp):
            ws[k] = w_t[i+1,val.astype(int)]/np.sum(w_t[i+1,M_temp.astype(int)])
        M_mat[i+1,:] = np.random.choice(M_temp,size = B,replace = True,p = ws)
        #LLs[i+1] = np.mean(w_t[i+1,M_mat[i+1,:].astype(int)])
    LL = np.sum(np.log(LLs))
    pool.close()
    return(LL,LLs,M_temp,w_t[-1,:],preds)  
