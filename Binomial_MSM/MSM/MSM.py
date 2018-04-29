# -*- coding: utf-8 -*-
import numpy as np
from Binomial_MSM.MSM.likelihood.original import likelihood
from Binomial_MSM.MSM.starting_val.starting_val import starting_values,starting_values_new
from Binomial_MSM.MSM.likelihood.new import likelihood_new,likelihood_pf
import scipy.optimize as opt
import pandas as pd
import time
#import matplotlib.pyplot as plt
import re
import statsmodels.api as sm
from Binomial_MSM.MSM.likelihood.original import T_mat_template

def MSM_original(data,kbar,startingvals):
        A_template = T_mat_template(kbar)
        startingvals, LLs,ordered_parameters = starting_values(data,startingvals,kbar,A_template)
        bnds = ((1.001,50),(1,1.99),(1e-3,0.999999),(1e-4,5))
        minimizer_kwargs = dict(method = "L-BFGS-B",bounds = bnds,args = (kbar,data,A_template,None))
        res = opt.basinhopping(likelihood,x0 = startingvals,minimizer_kwargs = minimizer_kwargs,
                               niter = 1)
        parameters,LL,niters,output = res.x,res.fun,res.nit,res.message
        print(LL,parameters)
        LL, LLs,pi_t = likelihood(parameters,kbar,data,A_template,None,2)
        LL = -LL
        
        return(LL,LLs,parameters,pi_t)

def MSM_new(data,kbar,startingvals,niter_basin =1):
        startingvals, LLs,ordered_parameters = starting_values_new(data,startingvals,kbar)
        bnds = ((1.001,50),(1,1.99),(1e-3,0.999999),(1e-4,5))
        minimizer_kwargs = dict(method = "L-BFGS-B",bounds = bnds,args = (kbar,data,None))
        print(startingvals)
        res = opt.basinhopping(likelihood_new,x0 = startingvals,minimizer_kwargs = minimizer_kwargs,
                               niter =niter_basin)
        parameters,LL,niters,output = res.x,res.fun,res.nit,res.message
        print(LL,parameters)
        #LL, LLs,pi_t,forward = likelihood_new(parameters,kbar,data,None,2)
        #LL = -LL
        
        #return(LL,LLs,parameters,pi_t,forward)
        return(LL,parameters)

def MSM_pf(data,kbar,startingvals,B):
        startingvals, LLs,ordered_parameters = starting_values_new(data,startingvals,kbar)
        bnds = ((1.001,50),(1,1.99),(1e-3,0.999999),(1e-4,5))
        minimizer_kwargs = dict(method = "L-BFGS-B",bounds = bnds,args = (kbar,dat,None))
        print(startingvals)
        res = opt.basinhopping(likelihood_new,x0 = startingvals,minimizer_kwargs = minimizer_kwargs,
                               niter =3)
        parameters,LL,niters,output = res.x,res.fun,res.nit,res.message
        
        LL,LLs,M_mat,w_t,preds = likelihood_pf(parameters,kbar,data,B)
        LL = -LL
        
        return(LL,LLs,M_mat,w_t,preds,parameters)
    
if __name__ == "__main__":
    """
    T = 1000
    kbar = 5
    g_kbar = 0.5
    b = 5
    m0 = 1.5
    m1 = 2-m0
    sig = 5/np.sqrt(252)
    g_s = np.zeros(kbar)
    M_s = np.zeros((kbar,T))
    g_s[0] = 1-(1-g_kbar)**(1/(b**(kbar-1)))
    for i in range(1,kbar):
        g_s[i] = 1-(1-g_s[0])**(b**(i))
    
    for j in range(kbar):
        M_s[j,:] = np.random.binomial(1,g_s[j],T)
    dat = np.zeros(T)
    tmp = (M_s[:,0]==1)*m1+(M_s[:,0]==0)*m0
    dat[0] = np.prod(tmp)
    for k in range(1,T):
        for j in range(kbar):
            if M_s[j,k]==1:
                tmp[j] = np.random.choice([m0,m1],1,p = [0.5,0.5])

        dat[k] = np.prod(tmp)
    dat = np.sqrt(dat)*sig* np.random.normal(size = T)
    dat = dat.reshape(-1,1)
    
    #dat = pd.read_csv("/Users/jan/Documents/Git_codes/Python_proj/data_demo.csv",header = None)
    #dat = np.array(dat)
    #dat.reshape(1,-1)[0] 
    """
    dat = pd.read_csv("/home/jan/Downloads/DEXJPUS.csv")
    for i,v in enumerate(dat.DATE):
        if re.sub("-.*","",v) == "1990":
            break
    
    dat = dat.loc[dat.DEXJPUS != "."].DEXJPUS.astype(float)
    dat = np.array(dat)
    dat = np.log(dat[1:])-np.log(dat[0:-1])
    #pdb.set_trace()
    dat = dat[:,np.newaxis]

    #train = dat[:i]
    #test = dat[i:]
    #start = time.time()
    #LL,LLs,params,pi = MSM_original(dat,5,None)
    #end =  time.time()
    #print(params[:-1],params[-1]/np.sqrt(252),LL,end-start)
    #start = time.time()
    #LL_,LLs_,params_,pi_,forward = MSM_new(dat,10,None)
    #end = time.time()
    #print(params_[:-1],params_[-1]/np.sqrt(252),LL_,end-start)
    start = time.time()
    #LL,params = MSM_new(dat[:i],5,None)
    LL,params = MSM_new(dat[:-30],10,None)
    params[-1] = params[-1]
    LL_,LLs_,M_mat,w_t,preds = likelihood_pf(params,10,dat,4000)
    LL,LLs,pi,forward = likelihood_new(params,10,dat,None,2)
    #L,Ls,M,W,preds,par = MSM_pf(dat,10,None,1000)
    end = time.time()
    print(end-start)
    #LL_,LLs_,pi_,forward = likelihood_new(params,10,dat,None,2)
    #Y = dat[i:]**2
    #X = sm.add_constant((params_[-1]/np.sqrt(252))**2*forward[i:])
    #model = sm.OLS(Y,X)
    #results = model.fit()
    #print(results.params)
    
    Y = dat[-30:]**2*100
    sigma = params[-1]*100/np.sqrt(252)
    #sigma = 0.25/np.sqrt(252)
    X = sm.add_constant((sigma*np.sqrt(preds[-30:]))**2)
    model = sm.OLS(Y,X)
    results = model.fit()
    print(results.params,results.rsquared)

    #
    #plt.scatter((params[-1]/np.sqrt(252)*np.sqrt(preds[-30:]))**2,dat[-30:]**2)
    #plt.xlim(0.00002,0.00006)
    #plt.ylim(0,0.00030)
    #mu,var = prediction(params_,5,pi_,len(test))
    #mu_,var_,m,w = prediction_pf(params_,5,1000,len(test),M,W)
    #for i,v in enumerate(mu):
    #    print(norm(v,np.sqrt(var[i])+1).pdf(test[i]))
    #    print(norm(v,np.sqrt(var_[i])+1).pdf(test[i]))
    
    #plt.plot(mu+2*np.sqrt(var))
    #plt.plot(dat[-5:])
    #lt.show()