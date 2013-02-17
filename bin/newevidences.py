#!/bin/python

"""PySUSY dataset manual evidence computation

This tool parses the results of PySUSY scans in the 'ev.dat' files and
reweights the likelihood data, and computes the evidence for the reweighted
data"""


import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import hdf5tools as l
import os

root='/media/backup1/xe_short_backups/CMSSMev-completed-runs/'

f = h5py.File(root+'CMSSMev.hdf5','r+') #CAREFUL! WRITE MODE ENABLED!

"""
dsetpaths = ['preLEP/log/mu+','preLEP/CCR/mu+',\
        'LEPXenon/log/mu+','LEPXenon/CCR/mu+',\
        'SUSY/log/mu+','SUSY/CCR/mu+',\
        'higgs/log/mu+','higgs/CCR/mu+']
"""
dsetpaths = ['higgs/log/mu+']
        
#access datasets in hdf5 file
dsets = [f[dsetpath]['evlive'] for dsetpath in dsetpaths] 

print 'Start loop'
for dsetIN in dsets:
    #---------get "no (g-2)" evidence------------------------------------------
    """
    fig= plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dsetIN['logL'][10000:])
    
    fig2= plt.figure()
    ax = fig2.add_subplot(111)
    ax.plot(dsetIN['neg2LogL'][10000:])
    
    plt.show()
    
    quit()
    """
    dset = dsetIN[:][['neg2LogL','logL','logw','deltaamuloglikelihood']]
    print dset['logL'][10:]
    print dset['deltaamuloglikelihood'][10:]
    
    logL = dset['logL'] - dset['deltaamuloglikelihood'] #subtract off (g-2) loglikelihood
    logw = dset['logw']
    print logw[10:]
    
    post = np.exp(logL + logw) #exp(logL + logw) = likelihood * prior (unnormalised posterior)
    print post[10:]
    
    Z2, dZ2, logZ2, dlogZ2 = l.computeevidence(post,logL,dsetIN.attrs['nlive'])     
    print logZ2
    dsetIN.attrs['no(g-2)logZ'] = [logZ2,dlogZ2] #Add to database     
    #-------------------------------------------------------------------





