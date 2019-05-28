#!/usr/bin/env python
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

nsim = 100
sig = 0.01
h = 67.36
#h = 74.03
nz = 30

h=np.loadtxt('H0_sklearn_sigh0p0'+str(int(sig*100))+'_'+str(nz)+'pts_h0p'+str(int(h*100))+'_SVM_ALL.dat')
#mae=np.loadtxt('mae_sklearn_sigh0p0'+str(int(sig*100))+'_'+str(nz)+'pts_h0p'+str(int(h*100))+'_EXT_ALL.dat')
#sc1=np.loadtxt('scoretrain_sklearn_sigh0p0'+str(int(sig*100))+'_'+str(nz)+'pts_h0p'+str(int(h*100))+'_ALL.dat')
#sc2=np.loadtxt('scoretest_sklearn_sigh0p0'+str(int(sig*100))+'_'+str(nz)+'pts_h0p'+str(int(h*100))+'_ALL.dat')

h_sort=np.sort(h)
#print(h_sort)
    
print( np.mean(h), np.std(h) )