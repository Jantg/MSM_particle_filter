# -*- coding: utf-8 -*-
import numpy as np
from Binomial_MSM.MSM.likelihood.sharedfunc import gofm
from numba import jit,float64, int64

def T_mat_template(kbar):
    """
    A function to generate transition matrix template
    
    kbar = number of multipliers in the model
    """
    kbar2 = 2**kbar
    A = np.zeros((kbar2,kbar2))
    for i in range(kbar2):
        for j in range(i,kbar2-i):
            A[i,j] = np.bitwise_xor(i,j)
    return(A)     


def transition_mat(A,inpt,kbar):
    """
    When given a template A and inputs this function will compute the 
    transition matrix A
    
    A = a template computed by T_mat_template function
    inpt = all parameters for MSM (b,m0,gamma_kbar,sigma in this order)
    kbar = number of multipliers in the model
    """
    # extract all necessary inputs from inpt and give name
    b = inpt[0]
    gamma_kbar = inpt[2]
    
    # compute gammas
    gamma = np.zeros((kbar,1))
    gamma[0,0] = 1 - (1 - gamma_kbar)**(1/(b**(kbar-1)))
    for i in range(1,kbar):
        gamma[i,0] = 1 - (1 - gamma[0,0])**(b**(i))
    # when when switching happes at k with gamma_k prob
    # new states are drawn half the time, other wise stays the same
    gamma = gamma*0.5
    gamma = np.c_[gamma,gamma]
    gamma[:,0] = 1 - gamma[:,0]
    
    #generate all combination of probabilities
    kbar2 = 2**kbar
    prob = np.ones((kbar2,1))
    for i in range(kbar2):
        for m in range(kbar):
            tmp = np.unpackbits(np.arange(i,i+1,dtype = np.uint16).view(np.uint8))
            tmp = np.append(tmp[8:],tmp[:8])
            prob[i,0] = prob[i,0] * gamma[kbar-m-1,tmp[-(m+1)]]
    A_ = np.zeros((kbar2,kbar2))
    for i in range(2**(kbar-1)):
        for j in range(i,2**(kbar-1)):
            A_[kbar2-i-1,j] = prob[np.rint(kbar2 - A.copy()[i,j]-1).astype(int),0]
            A_[kbar2-j-1,i] = A_[kbar2-i-1,j]
            A_[j,kbar2-i-1] = A_[kbar2-i-1,j]
            A_[i,kbar2-j-1] = A_[kbar2-i-1,j]
            A_[i,j] = prob[np.rint(A.copy()[i,j]).astype(int),0]
            A_[j,i] = A_[i,j]
            A_[kbar2-j-1,kbar2-i-1] = A_[i,j]
            A_[kbar2-i-1,kbar2-j-1] = A_[i,j]
        
    return(A_)

@jit(float64[:,:](float64[:,:],float64[:],int64))
def transition_mat_jit(A,inpt,kbar):
    """
    When given a template A and inputs this function will compute the 
    transition matrix A
    
    A = a template computed by T_mat_template function
    inpt = all parameters for MSM (b,m0,gamma_kbar,sigma in this order)
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
    prob = np.ones((kbar2,1))
    A_ = np.zeros((kbar2,kbar2))
    for i in range(kbar2):
        for m in range(kbar):
            tmp = np.unpackbits(np.arange(i,i+1,dtype = np.uint16).view(np.uint8))
            tmp = np.append(tmp[8:],tmp[:8])
            prob[i,0] =prob[i,0] * gamma[kbar-m-1,tmp[-(m+1)]]
    for i in range(2**(kbar-1)):
        for j in range(i,2**(kbar-1)):
            #pdb.set_trace()
            A_[kbar2-i-1,j] = prob[np.rint(kbar2 - A.copy()[i,j]-1).astype(int),0]
            A_[kbar2-j-1,i] = A_[kbar2-i-1,j]
            A_[j,kbar2-i-1] = A_[kbar2-i-1,j]
            A_[i,kbar2-j-1] = A_[kbar2-i-1,j]
            A_[i,j] = prob[np.rint(A.copy()[i,j]).astype(int),0]
            A_[j,i] = A_[i,j]
            A_[kbar2-j-1,kbar2-i-1] = A_[i,j]
            A_[kbar2-i-1,kbar2-j-1] = A_[i,j]
        
    return(A_)

@jit(float64[:,:](float64[:],int64))
def transition_mat_new_jit(inpt,kbar):
    """
    When given a template A and inputs this function will compute the 
    transition matrix A
    
    inpt = all parameters for MSM (b,m0,gamma_kbar,sigma in this order)
    kbar = number of multipliers in the model
    """

    b = inpt[0]
    gamma_kbar = inpt[2]
    gamma = np.zeros((kbar,1))
    gamma[0,0] = 1-(1-gamma_kbar)**(1/(b**(kbar-1)))
    for i in range(1,kbar):
        gamma[i,1] = 1-(1-gamma[0,0])**(b**(i))
    gamma = gamma*0.5
    gamma = np.c_[gamma,gamma]
    gamma[:,0] = 1 - gamma[:,1]
    kbar2 = 2**kbar
    A = np.zeros((kbar2,kbar2))
    prob = np.ones(kbar2)
    # all combination of probabilities from gamma_1*gamma_2*gamma_3, gamma_1*gamma_2*(1-gamma_3),...
    for i in range(kbar2):
        for m in range(kbar):
            tmp = np.unpackbits(np.arange(i,i+1,dtype = np.uint16).view(np.uint8))
            tmp = np.append(tmp[8:],tmp[:8])
            prob[i] =prob[i] * gamma[kbar-m-1,tmp[-(m+1)]]
    for i in range(kbar2):
        for j in range(kbar2):
            A[i,j] = prob[np.bitwise_xor(i,j)]
    return(A)
    
def likelihood(inpt,kbar,data,A_template,estim_flag,nargout = 1):
    """
    Computes the likelihood up to the end of the data
    depending on the number of inputs it will either return sum of daily log likelihood
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
    A = transition_mat(A_template,inpt,kbar)
    g_m = gofm(inpt,kbar)
    T = len(data)
    pi_mat = np.zeros((T+1,k2))
    LLs = np.zeros(T)
    pi_mat[0,:] = (1/k2)*np.ones(k2)
    pa = (2*np.pi)**(-0.5)
    s = sigma*g_m
    w_t = data
    w_t = pa*np.exp(-0.5*((w_t/s)**2))/s
    w_t = w_t + 1e-16
    
    for t in range(T):
        piA = np.dot(pi_mat[t,:],A)
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
        return(LL,LLs,pi_mat[-1,:])
    