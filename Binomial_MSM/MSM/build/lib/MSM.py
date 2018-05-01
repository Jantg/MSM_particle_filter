# -*- coding: utf-8 -*-
from Binomial_MSM.MSM.likelihood.original import likelihood
from Binomial_MSM.MSM.starting_val.starting_val import starting_values,starting_values_new
from Binomial_MSM.MSM.likelihood.new import likelihood_new,likelihood_pf
import scipy.optimize as opt
from Binomial_MSM.MSM.likelihood.original import T_mat_template

def MSM_original(data,kbar,startingvals):
    """
    Implementation of MSM close to the original one in Matlab

    data = data to use for likelihood calculation
    kbar = number of multipliers in the model
    startingvals = 4 parameters of the MSM, recommended to set to None so that the function can
    compute optimal starting values
    """
    A_template = T_mat_template(kbar)
    startingvals, LLs,ordered_parameters = starting_values(data,startingvals,kbar,A_template)
    bnds = ((1.001,50),(1,1.99),(1e-3,0.999999),(1e-4,5))
    minimizer_kwargs = dict(method = "L-BFGS-B",bounds = bnds,args = (kbar,data,A_template,None))
    res = opt.basinhopping(likelihood,x0 = startingvals,minimizer_kwargs = minimizer_kwargs,
                           niter = 1)
    parameters,LL,niters,output = res.x,res.fun,res.nit,res.message
    #print(LL,parameters)
    LL, LLs,pi_t = likelihood(parameters,kbar,data,A_template,None,2)
    LL = -LL
    
    return(LL,LLs,parameters,pi_t)

def MSM_new(data,kbar,startingvals,niter_basin =1):
    """
    Faster Implementation of MSM estimation

    data = data to use for likelihood calculation
    kbar = number of multipliers in the model
    startingvals = 4 parameters of the MSM, recommended to set to None so that the function can
    compute optimal starting values
    niter_basin = number of iterations for basinhopping, the greater the chance of getting a local minimum
    will decrease with the cost of computation time
    """

    startingvals, LLs,ordered_parameters = starting_values_new(data,startingvals,kbar)
    bnds = ((1.001,50),(1,1.99),(1e-3,0.999999),(1e-4,5))
    minimizer_kwargs = dict(method = "L-BFGS-B",bounds = bnds,args = (kbar,data,None))
    #print(startingvals)
    res = opt.basinhopping(likelihood_new,x0 = startingvals,minimizer_kwargs = minimizer_kwargs,
                           niter =niter_basin)
    parameters,LL,niters,output = res.x,res.fun,res.nit,res.message
    #print(LL,parameters)
    #LL, LLs,pi_t,forward = likelihood_new(parameters,kbar,data,None,2)
    #LL = -LL
    
    #return(LL,LLs,parameters,pi_t,forward)
    return(LL,parameters)
    
