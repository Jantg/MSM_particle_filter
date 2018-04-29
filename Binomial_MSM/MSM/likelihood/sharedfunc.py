# -*- coding: utf-8 -*-
import numpy as np

def gofm(inpt,kbar):
    """
    A function that calculates all possible volatility states
    """
    m0 = inpt[1]
    m1 = 2-m0
    kbar2 = 2**kbar
    g_m1 = np.arange(kbar2)
    g_m = np.zeros(kbar2)
    
    for i in range(kbar2):
        g =1
        for j in range(kbar):
            if np.bitwise_and(g_m1[i],(2**j))!=0:
                g = g*m1
            else:
                g = g*m0
        g_m[i] = g
    return(np.sqrt(g_m))