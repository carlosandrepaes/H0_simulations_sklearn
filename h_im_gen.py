#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import pickle

#hz as given by flat-LCDM model
def hz_model(x, a, b):
    return b*np.sqrt(a*(1.+x)**3. + (1.-a))

def sighz_fit(x, a, b):
    return a + b*x 

#z,hz,errhz,hzid = np.loadtxt('hz.dat', unpack='true')

nsim = 100

for n in range(nsim):

    #input fiducial Cosmology (P18 best-fit for TT,TE,EE+lowE+lensing)
    om = 0.3166
    sigom = 0.0084
    #h = 67.36
    #sigh = 0.54

    # h0 best-fit from Riess et al 2019 after LMC Cepheids inclusion
    h = 74.03
    sigh = 1.42

    # z-array
    zmin=0.10
    zmax=1.50
    nz=30
    z_arr=zmin+(zmax-zmin)*np.arange(nz)/(nz-1.0)

    sig = 0.01
    # hz values according to the fiducial Cosmology and the given z values
    #hz_arr=np.array([hz(z,om,h) for z in z_arr])
    #hz_sim=hz_model(z,om,h) + errhz*np.random.randn()

    # displaying results
    for i in range(nz):
        hz_arr = np.array([hz_model(z,om,h) + sig*hz_model(z,om,h)*np.random.randn() for z in z_arr])
        sighz_arr = hz_arr*sig
        hztype_arr = np.array([1 for z in z_arr])
        print(n, z_arr[i], hz_arr[i]/hz_model(z_arr[i],om,h))


    # saving the simulated hz results in a text file
    np.savetxt('input/hz_sim_sigh0p0'+str(int(sig*100))+'_'+str(nz)+'pts_h0p'+str(int(h*100))+'_mc#'+str(n+1)+'.dat', np.transpose([z_arr, hz_arr, sighz_arr, hztype_arr]))