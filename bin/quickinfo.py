#!/bin/python

"""Template script for quickly extracting small pieces of information
from the PySUSY CMSSM evidence project hdf5 databases"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import hdf5tools as l
import os

from guppy import hpy; h=hpy()


#PRELEP and LEP+XENON data
root='/media/backup1/temp_CMSSM_backup/pysusyAnalysis/CMSSMpoints/'
outdirroot='/home/farmer/Dropbox/Projects/pysusy_pytools/tmp_plots/xe/'

dirs = ['1preLEP','2preLEPCCR','3LEPXenon','4LEPXenonCCR']
dsets = ['preLEP/log/mu+/txt','preLEP/CCR/mu+/txt','LEPXenon/log/mu+/txt','LEPXenon/CCR/mu+/txt']

#SUSY and HIGGS data
#root='/media/backup1/xe_short_backups/CMSSMev-completed-runs/'
#outdirroot='/home/farmer/Dropbox/Projects/pysusy_pytools/tmp_plots/xe/'

#root='/media/backup1/temp_CMSSM_backup/pysusyAnalysis/CMSSMpoints/CMSSMproject_tmp/m2/'
#outdirroot='/home/farmer/Dropbox/Projects/pysusy_pytools/tmp_plots/m2/'

#dirs = ['5SUSY','6SUSYCCR','7higgs','8higgsCCR']
#dsets = ['SUSY/log/mu+/txt','SUSY/CCR/mu+/txt','higgs/log/mu+/txt','higgs/CCR/mu+/txt']
#dirs = ['5SUSY','6SUSYCCR','8higgsCCR']
#dsets = ['SUSY/log/mu+/txt','SUSY/CCR/mu+/txt','higgs/CCR/mu+/txt']
#dirs = ['SUSYCCR']
#dsets = ['SUSY/CCR/mu+/txt']

#-------------------------------------------------
# TASKS
#--------------------------------------------------

#----------FIND LOWEST HIGGS MASS OBTAINED IN PRE-LEP DATASET-----------
f = h5py.File(root+'CMSSMev.hdf5','r')
dset = f['preLEP/log/mu+/txt']

data = l.getcols(dset,['higgs','neg2LogL'])
print 'Minimum Higgs mass in dataset: ', min(data[:,0])

fig= plt.figure()
plt.plot(data[:,0],np.exp(-0.5*(data[:,1] - min(data[:,1]))),'.')
plt.show()






