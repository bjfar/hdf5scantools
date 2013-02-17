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

#Specify locations of datasets

#preLEP, LEPXenon, SUSY and higgs datasets
root1   = '/media/backup1/xe_short_backups/CMSSMev-completed-runs/'
#f1 = h5py.File(root1+'CMSSMev.hdf5','r+')   #BE CAREFUL, WRITE ACCESS GRANTED
f1 = h5py.File(root1+'CMSSMev.hdf5','r')

outdir = '/home/farmer/Dropbox/Projects/pysusy_pytools/tmp_plots/xe'   #save output plots here

#Specify datasets to be examined.
"""
dsets = {   '1preLEP'     : f1['preLEP/log/mu+'],
            '2LEP'        : f1['LEPXenon/log/mu+'],
            '3SUSY'       : f1['SUSY/log/mu+'],
            '4higgs'      : f1['higgs/log/mu+'],
            '5CCRpreLEP'  : f1['preLEP/CCR/mu+'],
            '6CCRLEP'     : f1['LEPXenon/CCR/mu+'],
            '7CCRSUSY'    : f1['SUSY/CCR/mu+'],
            '8CCRhiggs'   : f1['higgs/CCR/mu+']
        } 
"""        
dsets = {   '5CCRpreLEP'  : f1['preLEP/CCR/mu+'],
            '6CCRLEP'     : f1['LEPXenon/CCR/mu+'],
            '7CCRSUSY'    : f1['SUSY/CCR/mu+'],
            '8CCRhiggs'   : f1['higgs/CCR/mu+']
        } 

#dsets = { '4higgs'      : f1['higgs/log/mu+'],
#    }
chi2par = 'neg2LogL'
postpar = 'P'

"""
parslist = [['M0','M12'],\
            ['A0','TanB'],\
            ['higgs','BQloglikelihood']]
            
xylabs = [[r'$M_0$ (GeV)',r'$M_{1/2}$ (GeV)'],\
        [r'$A_0$ (GeV)',r'$\tan\beta$'],\
        [r'$m_h$ (GeV)',r'Eff. prior ($L$)']]
"""
        
parslist = [['higgs','BQloglikelihood']]
xylabs = [[r'$m_h$ (GeV)',r'$1/$Eff. prior']]

extpars = ['deltaamuloglikelihood','BQloglikelihood','MZout','muQ'] #extra parameters needed for reweighting

for dsetname,dset in dsets.items():
    print "Opening dataset {0}...".format(dset)
    #Create dataset analysis object
    allpars = list(set([x for y in parslist for x in y])) + [chi2par,postpar] + extpars

    dsetLINK = l.LinkDataSetForAnalysis(dset['evlive'],outdir+'/'+dsetname,allpars,chi2par,postpar)

    for noamu in [False,True]:
        #chi2 = 0
        #Dchi2 = 0
        #post = 0
        tag = ''    #no tag by default
        
        #reweight?
        if noamu:
            Dchi2 = -(-2*dsetLINK.dset['deltaamuloglikelihood']) #subtract off (g-2) chi^2
            tag = 'no(g-2)'
        else:
            Dchi2 = 0
        
        chi2 = dsetLINK.likedata + Dchi2
        chi2 = chi2 - np.min(chi2)  #rescale chi2 to minimum
        print dsetLINK.dset[postpar][-20:]
        postmod = np.exp(-0.5*Dchi2)
        if type(postmod)==type(np.array([])):  #only do this if Dchi2 contains a numpy array
            mask = np.isinf(postmod)
            print 'Removing infs from reweighted posterior. Number removed: ', np.sum(mask)
            postmod[mask] = 1    #remove infs that arise from previously highly excluded points 
                                        #(have loglikelihood of -300 or so and cause overflows in the exp)
                                        #These points may now be viable but because of the overflow we cannot
                                        #correctly reweight the posterior. There should not be many of these
                                        #so it should make little difference to the plots if we leave them
                                        #excluded
        post = dsetLINK.probdata*postmod
        print post[-20:]
        post = post / np.sum(post)  #re-normalise posterior
        
        for parsIN,xylab in zip(parslist,xylabs):
            #skip loop if trying to plot effective prior, and no effective prior was computed
            if (parsIN[0]=='BQloglikelihood' or parsIN[1]=='BQloglikelihood') and dsetLINK.CCRprior==False: continue
            
            dataALL = dsetLINK.getcols(parsIN+['MZout','muQ'])
            #convert effective prior loglikelihood to approx. fine tuning (1/Delta ~ likelihood penalty (from eff. prior))
            MZ = dataALL[:,2]
            muQ = dataALL[:,3]
            dataXY = dataALL[:,:2]
            for i in [0,1]:
                if (parsIN[i]=='BQloglikelihood'): 
                    cmu = 0.5 * (MZ / muQ) * (1 / np.exp(dataXY[:,i])) #approximate fine tuning factor 
                    #cmu = 1 / np.exp(dataXY[:,i])   #1 / total penalisation factor
                    dataXY[:,i] = cmu    #log10 of Delta
                    dsetLINK.make2Dlogscatterplot(dataXY,xylab,'{0}{1}{2}ev'.format(parsIN[0],parsIN[1],tag),dataL=chi2)
                                
            #Create and export plots
            #dsetLINK.make2Dscatterplot(dataXY,xylab,'{0}{1}{2}ev'.format(parsIN[0],parsIN[1],tag),dataL=chi2)
            #dsetLINK.make2Dprofileplot(dataXY,xylab,'{0}{1}{2}ev'.format(parsIN[0],parsIN[1],tag),dataL=chi2)
            #dsetLINK.make2Dposteriorplot(dataXY,xylab,'{0}{1}{2}ev'.format(parsIN[0],parsIN[1],tag),dataP=post)
            
            """
            #Also want to see the marginalised prior, out of curiosity.
            prior = dsetLINK.dset[logwpar]
            if dsetLINK.CCRprior: prior = prior + dsetLINK.dset['BQloglikelihood']
            prior = np.exp(prior)
            priorZ = np.sum(prior)
            prior = prior/priorZ    #normalise prior (shouldn't be needed if CCRprior not used, but won't hurt)
            dsetLINK.make2Dposteriorplot(dataXY,parsIN,'{0}{1}-prior'.format(parsIN[0],parsIN[1]),prior)
            """   



        
# OLD STUFF

quit() 
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
livePdata = np.exp(livelogL) * Pleft/len(livelogL)  # (unnormalised posterior for live points)
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
Z = sum(post)
logZ = np.log(Z)   #compute evidence
print 'logZ: ', logZ, sum(post/Z)
H = sum(post/Z * (logL - logZ))
dlogZ = np.sqrt(H/len(livelogL))   #error in logZ
dZ = max([np.exp(logZ + dlogZ) - Z, Z - np.exp(logZ - dlogZ)])
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

dataout = np.column_stack([data, post/Z]) #attach (normalised) posterior column to data points
print 't:', time.time()-t0

fig= plt.figure(figsize=plotsize)
ax = fig.add_subplot(111)
plot = l.margplot(ax,dataout,labels=pars[0:2])
cbar = fig.colorbar(plot)
#fig.savefig("{0}/{1}{2}marg.png".format(outdir,pars[0],pars[1]),dpi=(800/8))
#plt.close()

plt.show()






