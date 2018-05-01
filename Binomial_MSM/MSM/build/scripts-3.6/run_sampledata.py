# -*- coding: utf-8 -*-
from Binomial_MSM.MSM.MSM import MSM_new
import numpy as np

if __name__ == "__main__":
    T = 1000
    kbar = 3
    g_kbar = 0.8
    b = 6
    m0 = 1.6
    m1 = 2-m0
    sig = 2/np.sqrt(252)# we divide sigma by sqrt of the # of trading days
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
    LL,params = MSM_new(dat,3,None,1)
    print("real b=6,estimated b=",params[0],"\n",
          "real m0=1.6,estimated m0=",params[1],"\n",
          "real g_kbar=0.8,estimated g_kbar=",params[2],"\n",
          "real sig = 2, estimated sig=",params[3])