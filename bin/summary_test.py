#!/bin/python

"""PySUSY 'Evidence' project summary creator

The purpose of this tool is to collected information about the results 
of the PySUSY scans I have done for the CMSSM evidence project, collate 
them into an easily readble form, and help me know what has and has not 
been done. It will also be used to create summary information to be used
in the paper."""

import numpy as np
import h5py
import hdf5tools.tools as l

import matplotlib.pyplot as plt
import uncertainties as un
from uncertainties.umath import exp as uexp
plotsize=(11,6)

from math import floor, log10

#=======================================================================
# SETUP
#=======================================================================

#Specify locations of datasets containing evidences to extract

f1 = h5py.File('tests-19-06-12.hdf5','r')
outdir = ''   #save output plots here

#----AUGMENTED STANDARD MODEL EVIDENCE CHANGES------
# These are the evidence changes that are inflicted upon the REFERENCE MODEL
# Computed in "Dropbox/digitized_curves/SMmhOccam.py"

preLEP2LEPDev = 0.2840870694683335
LEP2ATLASDev = 0.046845360495

#Get evidences (and maximum loglikelihood) values to be compared
#(we are going to break up contributions to the Bayes factor into likelihood pieces
#and Occam factor pieces)

def getmax(dset,cols,exclude=None):
    """excude -- a value to skip over in the maximisation, e.g. a 
    default null value (perhaps zero!)"""
    ldata = l.getcols(dset,cols)
    #why is there an exp in here?!?!
    maxlogL = max([val for val in np.sum(ldata, axis=1) if val!=exclude]) #sum rows (likelihood contributions for each point) then find maximum
    return maxlogL
higgsXScols=['logl-XSgghaa','logl-XSgghWW2l2v','logl-XSgghZZ4l']

"""Make a plot to check out what the likelihood function is doing
data=l.getcols(f1['highm12ATLASHiggs']['timing'],['M0']+higgsXScols)
print data[:20]
fig = plt.figure(figsize=(18,8))
ax = fig.add_subplot(111)
ax.scatter(x=data[:,0], y=data[:,1], marker='.',s=1,lw=0,c='r')
ax.scatter(x=data[:,0], y=data[:,2], marker='.',s=1,lw=0,c='g')
ax.scatter(x=data[:,0], y=data[:,3], marker='.',s=1,lw=0,c='b')
ax.set_xlim(0,10000)
ax.set_ylim(top=0)
fig.savefig("higgslike.png",dpi=(800/8))
"""

#higgsmaxlogL=getmax(f1['higgs/log/mu+']['evlive'],higgsXScols)         #project 1 stuff
#CCRhiggsmaxlogL=getmax(f1['higgs/CCR/mu+']['evlive'],higgsXScols)
#highm12higgsmaxlogL=getmax(f1['highm12ATLASHiggs']['evlive'],higgsXScols)  #evlive dataset results
#LT2TeVhiggsmaxlogL=getmax(f1['LT2TeVATLASHiggs']['evlive'],higgsXScols)
#highm12higgsmaxlogL=getmax(f1['highm12ATLASHiggs']['timing'],higgsXScols,exclude=0.)   #timing dataset results have better statistics!
#LT2TeVhiggsmaxlogL=getmax(f1['LT2TeVATLASHiggs']['timing'],higgsXScols,exclude=0.)
#print highm12higgsmaxlogL
#print LT2TeVhiggsmaxlogL

#should be both the same, or pretty close to it.
#update: Righto, the highm12 data set found a better higgs point, interesting
highm12higgsmaxlogL=-0.660323
LT2TeVhiggsmaxlogL=-1.06858

#oh crap! I exponetiated these when I shouldn't have! the below are LIKELIHOOD
#values, not LOG likelihood values!!!
#higgsmaxlogL=0.374882   #to save time I have hardcoded in the results of the above 'getmax' calls.
#CCRhiggsmaxlogL=0.383291
#print higgsmaxlogL
#print CCRhiggsmaxlogL

#NOW, the COMBINED ATLAS likelihood function that we used to compute the SM higgs evidence change
#is normalised DIFFERENTLY to the THREE likelihood functions (one for each ATLAS search channel)
#that are used to compute the CMSSM higgs evidence change. We have to make these normalisations the
#SAME in order to compare their evidence changes fairly. Since the max(logL) value for the combined
#case is zero (min chi^2 = 0, because it is a delta chi^2 value), we rescale the CMSSM higgs likelihood
#function so that max(sum(higgsXScols))=0 also. We do this by scaling the evidence value directly, since
#the two are proportional.

#WAIT, CRAP. The CMSSM does NOT achieve anything like the possible maximum likelihood for the higgs search!
#We need to scale by the maximum likelihood value POSSIBLE in the combined higgs likelihood function (since this
#value is 1 in the pre-combined likelihood function used for the SM constraint), and THEN
#compute the likelihood ratio based on the maximum likelihood value ACTUALLY FOUND in the CMSSM scan.

combmaxlogL= -0.5*-7.4120989631 # -0.5 * minchi^2

#SIDE NOTE:
#This has the side effect of turning all the evidence changes listed below into simply "Occam factors", as
#I have defined them: O = DE / Lmax, where Lmax is the maximum likelihood value obtained for the new data
#which is changing the evidence. This is because the likelihood functions for the new data are all scaled
#to have values between zero and one (so Lmax=1), except the product of the ATLAS higgs likelihood functions
#(which although each component is scaled between zero and one, the product is not because the maxima are not
#aligned), which we have now rescaled to have this property.
#
#The likelihood ratio contribution to the evidence thus comes only from the SM ATLAS likelihood function, because
#the CMSSM has more freedom to fit the higgs signal than the SM, so the SM has a slighly lower maximum likelihood value.
#
#have now also scaled to be between zero and one
        
# 'name' : ([logZ,dlogZ], maxLogL)
dsetevs = { 'SM' : {
                'LEP'   : (un.ufloat([np.log(preLEP2LEPDev), 0.0000001]), 0),  
                'higgs'   : (un.ufloat([np.log(preLEP2LEPDev)+np.log(LEP2ATLASDev), 0.0000001]), -0.391223030156), #negative so counts as evidence FOR cMSSM
                },
            #-----------------SM VALUES DEALT WITH SEPERATELY-------------------
            'LT2TeV' : {
                'LEP'        : (un.ufloat(f1['LT2TeVLEPXENON']['evlive'].attrs['logZ']), 0),
                'higgs'      : (un.ufloat(f1['LT2TeVATLASHiggs']['evlive'].attrs['logZ']-np.array([combmaxlogL,0])), highm12higgsmaxlogL-combmaxlogL),
                },
            'highm12' : {
                'LEP'     : (un.ufloat(f1['highm12LEPXENON']['evlive'].attrs['logZ']), 0),
                'higgs'   : (un.ufloat(f1['highm12ATLASHiggs']['evlive'].attrs['logZ']-np.array([combmaxlogL,0])), LT2TeVhiggsmaxlogL-combmaxlogL),
                }
        }

#------------Data set checks--------------------------------------------
print 'SM OCCAM FACTOR:' # logZ - maxlogL -> O = (1/maxL) * Z
print np.log(LEP2ATLASDev)+0.391223030156, 1/np.exp(np.log(LEP2ATLASDev) - (-0.391223030156))
print 'SM/CMSSM maximum higgs likelihood ratio:'
print np.exp(-0.391223030156 - (highm12higgsmaxlogL-combmaxlogL))

print 'Evidences:'
for dk in dsetevs.keys():
    #print un2str(*dsetevs[d]) 
    print dk
    for key in dsetevs[dk].keys():
        print dsetevs[dk][key][0]

#------------Model testing specifications-------------------------------
datalist=['LEP','higgs']
#datachanges=[['preLEP->LEP','preLEP->SUSY','preLEP->higgs'],\
#            ['preLEP->LEP','LEP->SUSY','SUSY->higgs']]
altscenarios=['SM']   #alternate model (SM)
scenarios=['LT2TeV','highm12'] #CMSSM

#------------Do tests!--------------------------------------------------
l.evsummary(dsetevs,scenarios,altscenarios,datalist)
