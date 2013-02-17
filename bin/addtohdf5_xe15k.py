#!/bin/python

"""PySUSY CMSSM evidence project hdf5 filesystem interface

This tool is an interface for the PySUSY CMSSM evidence project hdf5
database. It is designed to make addition of data to the database
simpler (probably via bash scripts which run this tool)"""

import numpy as np
import h5py
import time
import csv
import hdf5tools as lib


#SUSY and HIGGS data
#---------Path to hdf5 database file------------------------------------
#roothdf5='/media/backup1/temp_CMSSM_backup/pysusyAnalysis/CMSSMpoints/CMSSMproject_tmp/xe/'
roothdf5='/media/backup1/xe_short_backups/CMSSMev-completed-runs/'
pathhdf5=roothdf5+'CMSSMev.hdf5'

#---------Common root directory of raw datasets-------------------------
#root='/media/backup1/temp_CMSSM_backup/pysusyAnalysis/CMSSMpoints/CMSSMproject_tmp/xe/'
root='/media/backup1/xe_short_backups/CMSSMev-completed-runs/'

#-------DATASETS TO IMPORT--------
#tuple contains ('target' hdf5 path,'source' linux path, datetime object, number of lines)
#Get the number of lines in each dataset first, using wc -l or similar
"""
datasets=[
('preLEP/log/mu+',root+'EV-PRELEP/EV-PRELEP-03:12:11','2011-12-3'),
('preLEP/CCR/mu+',root+'EV-PRELEP-CCR/EV-PRELEP-CCR-03:12:11','2011-12-3'),
('LEPXenon/log/mu+',root+'EV-FULL/EV-FULL-08:11:11','2011-11-8'),
('LEPXenon/CCR/mu+',root+'EV-FULL-CCR-B/EV-FULL-CCR-B-17:11:11','2011-11-17'),
('SUSY/log/mu+',root+'EVFULLLHC15k/EVFULLLHC15k-03:03:12','2012-03-08'),
('SUSY/CCR/mu+',root+'EVFULLCCRLHC15k/EVFULLCCRLHC15k-03:03:12','2012-03-08'),
('higgs/log/mu+',root+'EVFULLLHChiggs15k/EVFULLLHChiggs15k-03:03:12','2012-03-08'), #NOT YET FINISHED
('higgs/CCR/mu+',root+'EVFULLCCRLHChiggs15k/EVFULLCCRLHChiggs15k-03:03:12','2012-03-08'),
]
"""
datasets=[
('higgs/log/mu+',root+'EVFULLLHChiggs15k/EVFULLLHChiggs15k-03:03:12','2012-03-08'),
]


#Create new database
#
#print 'Creating hdf5 file {0}...'.format(pathhdf5)
#f = h5py.File(pathhdf5,'w')  #set to 'w' if you want to delete 
#                            #and recreate the whole thing, but BE CAREFUL
#f.close


#Create link object to database
dbase = lib.LinkDatabaseForImport(pathhdf5,'r+')

#Add 'txt' data to database
for dataset in datasets:
    dbase.importtxtdataset(*dataset,force=True,chunks=True)

#Add 'ev.dat' and live point data to database
for dataset in datasets:
    dbase.importevdataset(*dataset,force=True,chunks=True)

#Combine ev.dat and livep oint data into new database (usable as txt database)
for dataset in datasets:
     dbase.combineevlive(dataset[0],force=True)
    
quit()

"""
#------------------OLD STUFF-------------------------

dbase.importtxtdataset(*datasets[0],force=True,chunks=True)
dbase.importtxtdataset(*datasets[1],force=True,chunks=True)
#dbase.importtxtdataset(*datasets[2],force=True,chunks=True)
dbase.importtxtdataset(*datasets[3],force=True,chunks=True)


#Add 'ev.dat' data to database

dbase.importevdataset(*datasets[0],force=False,chunks=True)
dbase.importevdataset(*datasets[1],force=False,chunks=True)
#dbase.importevdataset(*datasets[2],force=False,chunks=True)
dbase.importevdataset(*datasets[3],force=False,chunks=True)



#Round 1 datasets and reweights
roothdf5='/media/backup1/temp_CMSSM_backup/pysusyAnalysis/CMSSMpoints/'
pathhdf5=roothdf5+'CMSSMev.hdf5'

print 'Creating hdf5 file {0}...'.format(pathhdf5)
f = h5py.File(pathhdf5,'w')  #set to 'w' if you want to delete 
                            #and recreate the whole thing, but BE CAREFUL
f.close

#Create link object to database
dbase = lib.linkdatabase(pathhdf5,'r+')
root='/media/backup1/temp_CMSSM_backup/pysusyAnalysis/CMSSMpoints/'

datasets=[
('preLEP/log/mu+',root+'EV-PRELEP-CCR/EV-PRELEP-CCR-03:12:11','2011-12-3'),

('preLEP/CCR/mu+',root+'EV-PRELEP/EV-PRELEP-03:12:11','2011-12-3'),

('LEPXenon/log/mu+',root+'EV-FULL/EV-FULL-08:11:11','2011-11-8'),
('LEPXenon/log/mu+/SUSY',root+'EV-FULL/EV-FULL-08:11:11-pars-LHC','2011-11-8'),
('LEPXenon/log/mu+/higgs',root+'EV-FULL/LHChiggsLISTtest-24:01:12','2012-01-24'),

('LEPXenon/CCR/mu+',root+'EV-FULL-CCR-B/EV-FULL-CCR-B-17:11:11','2011-11-17'),
('LEPXenon/CCR/mu+/SUSY',root+'EV-FULL-CCR-B/EV-FULL-CCR-B-17:11:11-pars-LHC','2011-11-17'),
('LEPXenon/CCR/mu+/higgs',root+'EV-FULL-CCR-B/EVFULLCCRBLHChiggs-27:01:12','2012-01-27')
]

for dataset in datasets:
    dbase.importtxtdataset(*dataset,force=True,chunks=True)
"""
