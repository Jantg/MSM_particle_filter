# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as opt
from Binomial_MSM.MSM.likelihood.original import likelihood
from Binomial_MSM.MSM.likelihood.new import likelihood_new


def starting_values(data,startingvals,kbar,A_template):
    """
    Computes the optimal starting values given original likelihood function
    
    data = data to use for likelihood calculation
    startingvals = if already supplied, this function just returns those values
    kbar = number of multipliers in the model
    A_template = template for the transition matrix (see T_mat_template function)
    """
    if not isinstance(startingvals,list):
        startingvals = [startingvals]
    if None in startingvals:
        print("No starting values entered: Using grid-search")
        
        b = np.array([1.5,5,15,30])
        lb = len(b)
        g = np.array([.1,.5,.9,.95])
        lg = len(g)
        sigma = np.std(data)*np.sqrt(252)
        output_parameters = np.zeros(((lb*lg),3))
        LLs = np.zeros((lb*lg))
        m0_lower = 1.2
        m0_upper = 1.8
        idx = 0
        for i in range(lb):
            for j in range(lg):
                xopt,fval,ierr,numfunc = opt.fminbound(likelihood,
                                                 x1 = m0_lower,x2 = m0_upper,xtol = 1e-6,
                                                 args = (kbar,data,A_template,[b[i],g[j],sigma]),full_output = True)
                m0,LL = xopt,fval
                output_parameters[idx,:] = b[i],m0,g[j]
                LLs[idx] = LL
                idx +=1
        idx = np.argsort(LLs)
        LLs = np.sort(LLs)
        startingvals = output_parameters[idx[0],:].tolist()+[sigma]
        output_parameters = output_parameters[idx,:]
        return(startingvals,LLs,output_parameters)
    elif len(startingvals) != 4:
        print("4 starting values are required,specify all or set those to None")
    else:
        return(startingvals)

def starting_values_new(data,startingvals,kbar):
    """
    Computes the optimal starting values given new likelihood function
    
    data = data to use for likelihood calculation
    startingvals = if already supplied, this function just returns those values
    kbar = number of multipliers in the model
    """    
    if not isinstance(startingvals,list):
        startingvals = [startingvals]
    if None in startingvals:
        print("No starting values entered: Using grid-search")
        
        b = np.array([1.5,5,15,30])
        lb = len(b)
        g = np.array([.1,.5,.9,.95])
        lg = len(g)
        sigma = np.std(data)*np.sqrt(252)
        output_parameters = np.zeros(((lb*lg),3))
        LLs = np.zeros((lb*lg))
        m0_lower = 1.2
        m0_upper = 1.8
        idx = 0
        for i in range(lb):
            for j in range(lg):
                xopt,fval,ierr,numfunc = opt.fminbound(likelihood_new,
                                                 x1 = m0_lower,x2 = m0_upper,xtol = 1e-6,
                                                 args = (kbar,data,[b[i],g[j],sigma]),full_output = True)
                m0,LL = xopt,fval
                output_parameters[idx,:] = b[i],m0,g[j]
                LLs[idx] = LL
                idx +=1
        idx = np.argsort(LLs)
        LLs = np.sort(LLs)
        startingvals = output_parameters[idx[0],:].tolist()+[sigma]
        output_parameters = output_parameters[idx,:]
        return(startingvals,LLs,output_parameters)
    elif len(startingvals) != 4:
        print("4 starting values are required,specify all or set those to None")
    else:
        return(startingvals)
