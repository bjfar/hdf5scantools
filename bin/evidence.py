#!/bin/python

"""PySUSY dataset manual evidence computation

This tool parses the results of PySUSY scans in the 'ev.dat' files and
recomputes the evidence manually based on the data therein. This allows
estimates of the uncertainties in the evidence of 'reweighted' scans to
be computed."""


import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import hdf5tools as l
import os

#root='/media/backup1/temp_CMSSM_backup/pysusyAnalysis/CMSSMpoints/CMSSMproject_tmp/xe/'
root='/media/backup1/xe_short_backups/CMSSMev-completed-runs/'
outdirroot='/home/farmer/Dropbox/Projects/pysusy_pytools/tmp_plots/xe/'

f = h5py.File(root+'CMSSMev.hdf5','r')

dirs = ['1preLEP','2preLEPCCR','3LEPXenon','4LEPXenonCCR',\
    '5SUSY','6SUSYCCR','7higgs','8higgsCCR']
dsets = ['preLEP/log/mu+','preLEP/CCR/mu+',\
        'LEPXenon/log/mu+','LEPXenon/CCR/mu+',\
        'SUSY/log/mu+','SUSY/CCR/mu+',\
        'higgs/log/mu+','higgs/CCR/mu+']

#dirs = []
#dsets = ['SUSY/log/mu+/txt','SUSY/CCR/mu+/txt','higgs/log/mu+/txt','higgs/CCR/mu+/txt']

#dsets = ['SUSY/log/mu+','SUSY/CCR/mu+','higgs/log/mu+','higgs/CCR/mu+']
#dsets = ['SUSY/CCR/mu+']
#dsets = ['SUSY/log/mu+']
#dsets = ['preLEP/log/mu+','preLEP/CCR/mu+']

#dirs = ['SUSY','SUSYCCR','higgs','higgsCCR']
#dirs = ['5SUSY','6SUSYCCR','8higgsCCR']
#dirs = ['6SUSYCCR']
#dirs = ['5SUSY']
#dirs = ['1preLEP','2preLEPCCR']

outdirs = [outdirroot+diri for diri in dirs] 

loglpar = 'logL'
logwpar = 'logw'

allpars = ['M0','M12'] + ['BQloglikelihood','deltaamuloglikelihood'] + ['mode'] + [loglpar, logwpar]

for outdir,dsetpath in zip(outdirs,dsets):
    print "Opening dataset {0}...".format(dsetpath)
    #Create dataset analysis object
    dsetLINK = l.LinkEvDataSetForAnalysis(f[dsetpath]['ev'],f[dsetpath]['live'],outdir,allpars,loglpar,logwpar)
    
    parsIN = ['M0','M12']

    dataXY = dsetLINK.getcols(parsIN)

    #Create and export plots
    dsetLINK.make2Dscatterplot(dataXY,parsIN,'{0}{1}ev'.format(*parsIN))
    dsetLINK.make2Dprofileplot(dataXY,parsIN,'{0}{1}ev'.format(*parsIN))
    dsetLINK.make2Dposteriorplot(dataXY,parsIN,'{0}{1}ev'.format(*parsIN))
    """DOESN'T REALLY WORK VERY WELL
    #Also want to see the marginalised prior, out of curiosity.
    prior = dsetLINK.dset[logwpar]
    if dsetLINK.CCRprior: prior = prior + dsetLINK.dset['BQloglikelihood']
    prior = np.exp(prior)
    priorZ = np.sum(prior)
    prior = prior/priorZ    #normalise prior (shouldn't be needed if CCRprior not used, but won't hurt)
    dsetLINK.make2Dposteriorplot(dataXY,parsIN,'{0}{1}-prior'.format(parsIN[0],parsIN[1]),prior)
        """                
    continue
    
    #--------------TEST REWEIGHTING--------------------
    # try removal of (g-2)
    print 'Removing (g-2) loglikelihood compenent...'
    evamulogL = evdset['deltaamuloglikelihood']
    liveamulogL = livedset['deltaamuloglikelihood']
    
    evlogL = evlogL - evamulogL     #subtract g-2 log-likelihood
    livelogL = livelogL - liveamulogL
    logL = np.append(evlogL,livelogL)
        
    evPdata = np.exp(evlogL + evlogw) #exp(logL + logw) = likelihood * prior (unnormalised posterior)
    Pleft = 1 - sum(np.exp(evlogw))
    print 'Pleft: ',Pleft, sum(np.exp(evlogw))
    if Pleft < 0: Pleft = 0     #Negative prior volume left makes no sense, so assume that correct amount left is negligible.
    nlive = len(livelogL)
    livePdata = np.exp(livelogL) * Pleft/nlive  # (unnormalised posterior for live points)
    post = np.append(evPdata,livePdata) #stitch posterior pieces together
    """
    #slice data to simulate new cut
    select = np.bitwise_or(data[:,0]>1000, data[:,1]>1000) #get selection mask
    print select[0:20]
    print select[-20::]
    data = data[select]
    post = post[select]
    logL = logL[select]
    
    print post[-20::]
    print len(select), len(post)
    """
    #---EVIDENCE COMPUTATIONS-------
    Z, dZ, logZ, dlogZ = computeevidence(post,logL,nlive)

    print 'error (dlogZ, dlogZ/logZ, dZ/Z ): ', dlogZ, dlogZ/logZ,  dZ/Z    #dlogZ = dZ/Z for small dlogZ.
    Z2 = Z
    logZ2 = logZ
    dlogZ2 = dlogZ
    dZ2 = dZ
    #------------------------------
    # Compute evidence ratio and error in ratio
    
    dZ1Z2 = max(np.abs([
    (Z1+dZ1)/(Z2+dZ2) - Z1/Z2,
    (Z1-dZ1)/(Z2+dZ2) - Z1/Z2,
    (Z1+dZ1)/(Z2-dZ2) - Z1/Z2,
    (Z1-dZ1)/(Z2-dZ2) - Z1/Z2,
    ]))
    print dZ1Z2/(Z1/Z2), dZ1/Z1 + dZ2/Z2
    print 'Evidence ratio: {0:5f} +/- {1:5f}'.format(Z1/Z2, dZ1Z2)
    
    #Create and export plots
    dsetLINK.make2Dscatterplot(dataXY,parsIN,'{0}{1}ev-(g-2)'.format(*parsIN),dataL=-2*logL)
    dsetLINK.make2Dprofileplot(dataXY,parsIN,'{0}{1}ev-(g-2)'.format(*parsIN),dataL=-2*logL)
    dsetLINK.make2Dposteriorplot(dataXY,parsIN,'{0}{1}ev-(g-2)'.format(*parsIN),dataP=post/Z)






