#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:07:34 2018

@author: root
"""

import numpy as np
import scipy
from MSM_likelihood_original import MSM_likelihood
from starting_vals_original import MSM_starting_values
import pandas as pd

def T_mat_template(kbar):
    kbar2 = 2**kbar
    A = np.zeros((kbar2,kbar2))
    for i in range(kbar2):
        for j in range(i,kbar2-i):
            A[i,j] = np.bitwise_xor(i,j)
    return(A)     
def MSM_modified(data,kbar,startingvals):
        A_template = T_mat_template(kbar)
        startingvals, LLs,ordered_parameters = MSM_starting_values(data,startingvals,kbar,A_template)
        bnds = ((1,50),(1,1.99),(1e-3,0.999999),(1e-4,5))
        minimizer_kwargs = dict(method = "L-BFGS-B",bounds = bnds,args = (kbar,dat,A_template,None))
        res = scipy.optimize.basinhopping(MSM_likelihood,x0 = startingvals,minimizer_kwargs = minimizer_kwargs,niter = 3)
        parameters,LL,niters,output = res.x,res.fun,res.nit,res.message
        LL, LLs = MSM_likelihood(parameters,kbar,data,A_template,None,2)
        LL = -LL
        
        return(LL,LLs,parameters)

if __name__ == "__main__":
    dat = pd.read_csv("data_demo.csv",header = None)
    dat = np.array(dat)
    LL_,LLs_,params_= MSM_modified(dat,3,None)        
    
