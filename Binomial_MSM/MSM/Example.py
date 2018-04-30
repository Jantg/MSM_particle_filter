# -*- coding: utf-8 -*-
"""
30-step ahed prediction with particle filter using USD/JPY exchange data
"""

import pandas as pd
import re
import numpy as np
from Binomial_MSM.MSM.MSM import MSM_new
from Binomial_MSM.MSM.likelihood.new import likelihood_pf
from Binomial_MSM.MSM.prediction.predict import prediction_pf
import matplotlib.pyplot as plt
# read the data in the current repository
dat = pd.read_csv("DEXJPUS.csv")
for i,v in enumerate(dat.DATE):
    if re.sub("-.*","",v) == "1990":
        break

dat = dat.loc[dat.DEXJPUS != "."].DEXJPUS.astype(float)
dat = np.array(dat)
dat = np.log(dat[1:])-np.log(dat[0:-1])
dat = dat[:,np.newaxis]
# conduct MSM estimation
LL,params = MSM_new(dat[:i],5,None)
# simulate states up to the most recent data
LL_,LLs_,M_mat,w_t,preds = likelihood_pf(params,5,dat[:i],1000)
#predict the 30 step ahead states 
mu_,var_,par,wei = prediction_pf(params,10,1000,30,M_mat,w_t)

plt.plot(mu_,color = "blue",label = "predicted state of the volatility")
plt.plot(dat[i:i+30]**2,color = "red",label = "real volatility")
plt.legend()
plt.show()