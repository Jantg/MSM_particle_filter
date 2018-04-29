# -*- coding: utf-8 -*-
import pdb
import numpy as np
from Binomial_MSM.MSM.likelihood.new import transition_mat_new,transition_prob
from Binomial_MSM.MSM.likelihood.sharedfunc import gofm
import multiprocessing
from multiprocessing import Pool
from functools import partial

def prediction(inpt,kbar,pi_t,n_steps):
    pi_n = np.zeros((n_steps,pi_t.shape[0]))
    g_m = gofm(inpt,kbar)
    sigma = inpt[3]/np.sqrt(252)
    A = transition_mat_new(inpt,kbar)
    pred_means = np.zeros(n_steps)
    pred_vars = np.zeros(n_steps)
    for i in range(n_steps):
        A_n = np.array(np.matrix(A)**(i+1))
        pi_n[i,:] = np.dot(pi_t,A_n)
        pred = (sigma)**2*(g_m)
        pred_means[i] = np.average(pred,weights = pi_n[i,:])
        pred_vars[i] = np.average((pred-pred_means[i])**2,weights = pi_n[i,:])
    return(pred_means,pred_vars)

def sim_one_step(M,inpt,kbar,Ms):
    next_state = []
    for i,v in enumerate(M):
        probs = transition_prob(tuple(inpt),v,kbar)[0]
        next_state.append(np.random.choice(Ms,size = 1, p = probs))
    return(next_state)
        
def prediction_pf(inpt,kbar,B,n_steps,particles,weights):
    g_m = gofm(inpt,kbar)
    sigma = inpt[3]/np.sqrt(252)
    M_pred = np.zeros((n_steps+1,B))
    Ms = np.arange(len(g_m))
    M_pred[0,:] = particles
    pred_means = np.zeros(n_steps)
    pred_vars = np.zeros(n_steps)
    cpu_count = multiprocessing.cpu_count()
    #pdb.set_trace()
    len_part = B/cpu_count
    for i in range(n_steps):
        M_parts= [M_pred[i,int(j*len_part):int((j+1)*len_part)] for j in range(cpu_count)]
        
        pool = Pool(processes = cpu_count)
        #pdb.set_trace()
        M_pred[i,:] = np.concatenate(pool.map(partial(sim_one_step,inpt = inpt,
              kbar = kbar,Ms = Ms),M_parts)).ravel()
        pool.close()
        #for j,val in enumerate(M_pred[i,:]):
        #    probs = transition_prob(tuple(inpt),val,kbar)[0]
        #    M_pred[i+1,j] = np.random.choice(Ms,size = 1, p = probs)
        pred_particles = (sigma)**2*(g_m[M_pred[i+1,:].astype(int)])
        #pdb.set_trace()
        w =  weights[M_pred[i+1,:].astype(int)]
        w = w/np.sum(w)
        pred_means[i] = np.average(pred_particles,weights=w)
        pred_vars[i] = np.average((pred_particles-pred_means[i])**2,weights=w)
    return(pred_means,pred_vars,M_pred,weights)  