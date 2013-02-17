#!/bin/python

"""PySUSY 'Evidence' project summary creator

The purpose of this tool is to collected information about the results 
of the PySUSY scans I have done for the CMSSM evidence project, collate 
them into an easily readble form, and help me know what has and has not 
been done. It will also be used to create summary information to be used
in the paper."""

#import copy
import numpy as np
#from uncertainties import ufloat
#from uncertainties.umath import *
#import decimal as d
import h5py
import hdf5tools as l

import matplotlib.pyplot as plt
import uncertainties as un
from uncertainties.umath import exp as uexp
plotsize=(11,6)

#--------------useful functions-------------------------
from math import floor, log10

def column(matrix, i):
    return [row[i] for row in matrix]
    
# uncertainty to string
def un2str(x, xe, precision=2):
    """pretty print nominal value and uncertainty

    x  - nominal value
    xe - uncertainty
    precision - number of significant digits in uncertainty

    returns shortest string representation of `x +- xe` either as
        x.xx(ee)e+xx
    or as
        xxx.xx(ee)"""
    # base 10 exponents
    x_exp = int(floor(log10(x)))
    xe_exp = int(floor(log10(xe)))

    # uncertainty
    un_exp = xe_exp-precision+1
    un_int = round(xe*10**(-un_exp))

    # nominal value
    no_exp = un_exp
    no_int = round(x*10**(-no_exp))

    # format - nom(unc)exp
    fieldw = x_exp - no_exp
    fmt = '%%.%df' % fieldw
    result1 = (fmt + '(%.0f)e%d') % (no_int*10**(-fieldw), un_int, x_exp)

    # format - nom(unc)
    fieldw = max(0, -no_exp)
    fmt = '%%.%df' % fieldw
    result2 = (fmt + '(%.0f)') % (no_int*10**no_exp, un_int*10**max(0, un_exp))

    # return shortest representation
    if len(result2) <= len(result1):
        return result2
    else:
        return result1

#=======================================================================
# SETUP
#=======================================================================

#Specify locations of datasets containing evidences to extract

#preLEP, LEPXenon, SUSY and higgs datasets
root1   = '/media/backup1/xe_short_backups/CMSSMev-completed-runs/'
f1 = h5py.File(root1+'CMSSMev.hdf5','r')

outdir = '/home/farmer/Dropbox/Projects/pysusy_pytools/tmp_plots'   #save output plots here

#----AUGMENTED STANDARD MODEL EVIDENCE CHANGES------
# These are the evidence changes that are inflicted upon the REFERENCE MODEL
# Computed in "Dropbox/digitized_curves/SMmhOccam.py"

preLEP2LEPDev = 0.2840870694683335
LEP2ATLASDev = 0.046845360495

#Get evidences (and maximum loglikelihood) values to be compared
#(we are going to break up contributions to the Bayes factor into likelihood pieces
#and Occam factor pieces)

def getmax(dset,cols):
    ldata = l.getcols(dset,cols)
    maxlogL = max(np.exp(np.sum(ldata, axis=1))) #sum rows (likelihood contributions for each point) then find maximum
    return maxlogL
higgsXScols=['XShaaloglikelihood','XShWW2l2vloglikelihood','XShZZ4lloglikelihood']

#higgsmaxlogL=getmax(f1['higgs/log/mu+']['evlive'],higgsXScols)
#CCRhiggsmaxlogL=getmax(f1['higgs/CCR/mu+']['evlive'],higgsXScols)
#should be both the same, or pretty close to it.

higgsmaxlogL=0.374882   #to save time I have hardcoded in the results of the above 'getmax' calls.
CCRhiggsmaxlogL=0.383291
print higgsmaxlogL
print CCRhiggsmaxlogL
        
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
            'preLEP'  : (un.ufloat([0,0]), 0.0000001),
            'LEP'   : (un.ufloat([np.log(preLEP2LEPDev), 0.0000001]), 0),  
            'SUSY'   : (un.ufloat([np.log(preLEP2LEPDev), 0.0000001]), 0),
            'higgs'   : (un.ufloat([np.log(preLEP2LEPDev)+np.log(LEP2ATLASDev), 0.0000001]), -0.391223030156), #negative so counts as evidence FOR cMSSM
            },
            #-----------------SM VALUES DEALT WITH SEPERATELY-------------------
            'log' : {
            'preLEP'     : (un.ufloat(f1['preLEP/log/mu+']['evlive'].attrs['logZ']), 0),
            'LEP'        : (un.ufloat(f1['LEPXenon/log/mu+']['evlive'].attrs['logZ']), 0),
            'SUSY'       : (un.ufloat(f1['SUSY/log/mu+']['evlive'].attrs['logZ']), 0),
            'higgs'      : (un.ufloat(f1['higgs/log/mu+']['evlive'].attrs['logZ']-np.array([combmaxlogL,0])), higgsmaxlogL-combmaxlogL),
            },
            'CCR' : {
            'preLEP'  : (un.ufloat(f1['preLEP/CCR/mu+']['evlive'].attrs['logZ']), 0),
            'LEP'     : (un.ufloat(f1['LEPXenon/CCR/mu+']['evlive'].attrs['logZ']), 0),
            'SUSY'    : (un.ufloat(f1['SUSY/CCR/mu+']['evlive'].attrs['logZ']), 0),
            'higgs'   : (un.ufloat(f1['higgs/CCR/mu+']['evlive'].attrs['logZ']-np.array([combmaxlogL,0])), CCRhiggsmaxlogL-combmaxlogL),
            },
            'no(g-2)log' : {
            'preLEP'     : (un.ufloat(f1['preLEP/log/mu+']['evlive'].attrs['no(g-2)logZ']), 0),
            'LEP'        : (un.ufloat(f1['LEPXenon/log/mu+']['evlive'].attrs['no(g-2)logZ']), 0),
            'SUSY'       : (un.ufloat(f1['SUSY/log/mu+']['evlive'].attrs['no(g-2)logZ']), 0),
            'higgs'      : (un.ufloat(f1['higgs/log/mu+']['evlive'].attrs['no(g-2)logZ']-np.array([combmaxlogL,0])), higgsmaxlogL-combmaxlogL),
            },
            'no(g-2)CCR' : {
            'preLEP'  : (un.ufloat(f1['preLEP/CCR/mu+']['evlive'].attrs['no(g-2)logZ']), 0),
            'LEP'     : (un.ufloat(f1['LEPXenon/CCR/mu+']['evlive'].attrs['no(g-2)logZ']), 0),
            'SUSY'    : (un.ufloat(f1['SUSY/CCR/mu+']['evlive'].attrs['no(g-2)logZ']), 0),
            'higgs'   : (un.ufloat(f1['higgs/CCR/mu+']['evlive'].attrs['no(g-2)logZ']-np.array([combmaxlogL,0])), CCRhiggsmaxlogL-combmaxlogL)
            }
        }

print 'SM OCCAM FACTOR:' # logZ - maxlogL -> O = (1/maxL) * Z
print np.log(LEP2ATLASDev)+0.391223030156, 1/np.exp(np.log(LEP2ATLASDev) - (-0.391223030156))
print 'SM/CMSSM maximum higgs likelihood ratio:'
print np.exp(-0.391223030156 - (higgsmaxlogL-combmaxlogL))


datalist=['preLEP','LEP','SUSY','higgs']
datachanges=[['preLEP->LEP','preLEP->SUSY','preLEP->higgs'],\
            ['preLEP->LEP','LEP->SUSY','SUSY->higgs']]

print 'Evidences:'
for dk in dsetevs.keys():
    #print un2str(*dsetevs[d]) 
    print dk
    for key in dsetevs[dk].keys():
        print dsetevs[dk][key][0]
            
#This error estimate is very naive and intended only as a rough order of
#magnitude guide.
def DlogevstoDev((logev0,dlogev0),(logev1,dlogev1)):
    #change two logevidence and uncert.'s to Delta(evidence) and uncert.
    Dlogev = logev1 - logev0
    Dev = np.exp(Dlogev)
    #take difference of extremes of the error 
    pmlogev1s = [(logev1+dlogev1),(logev1-dlogev1)]
    pmlogev0s = [(logev0+dlogev0),(logev0-dlogev0)]
    #find maximum error caused by using these extreme values and call that
    #the approximate standard error
    dDev = max([abs(np.exp(pmlogev1-pmlogev0) - Dev) \
            for pmlogev1 in pmlogev1s for pmlogev0 in pmlogev0s])
    dinvDev = max([1/abs(np.exp(pmlogev1-pmlogev0) - 1/Dev) \
            for pmlogev1 in pmlogev1s for pmlogev0 in pmlogev0s])
    return Dev, (dDevupper,dDevlower), dinvDev
    
def computeBayesFact((H0evA,H0evB,H0maxlogL),(H1evA,H1evB,H1maxlogL)):
    """Computes Bayes factors for a change of data, by comparing evidence
    before and after the data changed between two hypotheses
    Args
    H0evA - Initial log evidence for hypothesis 0
    H0evB - Final log evidence for hypothesis 0 (after new data added)
    H0maxlogL - Maximum value of the loglikelihood function for the new data achieved by H0
    H1evA - Initial log evidence for hypothesis 1
    H1evB - Final log evidence for hypothesis 1 (after new data added)
    H1maxlogL - Maximum value of the loglikelihood function for the new data achieved by H1
    EVIDENCE VALUES ARE TUPLE PAIRS OF THE FORM (VALUE,UNCERTAINTY)
    """
    
    #Compute log evidence change for H0 and H1
    DevH0 = H0evB - H0evA 
    DevH1 = H1evB - H1evA
    #print DevH0, DevH1
    #print dir(H0evB)
    #print dir(un.ufloat(DevH0))
    
    #want to add uncertainties directly (rather than in quadrature) because there 
    #will be odd correlations I expect and we should be conservative. This was not
    #done automatically so we shall modify the uncertainty manually.
    #FORGET THIS FOR NOW, IS BEING A PAIN
    #DevH0.__setattr__(H0evB.std_dev() + H0evA.std_dev())
    #DevH1.set_std_dev(H1evB.std_dev() + H1evA.std_dev())

    #print 'DevH0', DevH0
    #print 'DevH1', DevH1
    
    #Compute Bayes factor for H0:H1 model comparison
    B = DevH0 - DevH1
    #B.set_std_dev(DevH0.std_dev() + DevH1.std_dev())
        
    #Compute maximum (log)likelihood ratio (for new data only)
    LR = H0maxlogL - H1maxlogL
    
    #Compute individual (log)Occam factors for H0 and H1
    OH0 = DevH0 - H0maxlogL
    OH1 = DevH1 - H1maxlogL
    
    #Check that product (log sum) of Occam factors and likelihood ratio equal the Bayes factor
    #print 'Bayes factor:', B
    #print 'Occam factors (H0,H1):', OH0, OH1
    #print 'LR:', LR, 'Sum:', OH0[0] - OH1[0] + LR   #OH1 needs to be negative since it is in the denominator of the Bayes factor

    return B, LR, OH0, OH1
    
print '----------------------------------------------------------------'
print 'Belief change factors'
print '----------------------------------------------------------------'

altscenarios=['SM']   #alternate model (SM)
scenarios=['log','CCR','no(g-2)log','no(g-2)CCR'] #CMSSM

#build results dictionaries
#there is a results dictionary for each scenario vs alternate scenario dataset pair
Bresults={}
for Halt in altscenarios:
    Bresults[Halt] = {}
    for H0 in scenarios:
        Bresults[Halt][H0] = {}
        
for i,Halt in enumerate(altscenarios):
    for j,H0 in enumerate(scenarios):
        print '------{0} VS {1}---------'.format(Halt,H0)
        #'initial' dataset key (name)
        dkinit = datalist[0]
        for k,dk in enumerate(datalist):
            #comparison with 'initial' dataset
            if k==0: continue #(skip initial dataset since we are comparing to it)

            H0evA = dsetevs[Halt][dkinit][0]
            H0evB = dsetevs[Halt][dk][0]
            H0maxlogL = dsetevs[Halt][dk][1]
            H1evA = dsetevs[H0][dkinit][0]
            H1evB = dsetevs[H0][dk][0]
            H1maxlogL = dsetevs[H0][dk][1]
            B, LR, OH0, OH1 = computeBayesFact((H0evA,H0evB,H0maxlogL),(H1evA,H1evB,H1maxlogL)) #returns B, LR, OH0, OH1
            Bresults[Halt][H0]['{0}->{1}'.format(dkinit,dk)] = (B, LR, OH0, OH1)
            #print B[0], B[1]
            #print un2str(np.exp(B[0]),np.exp(B[0])*B[1],2)
            #print uexp(B).nominal_value, uexp(B).std_dev()
            print '{0:<10}-->{1:<10} ; B:{2}, OH0:{3}, OH1:{4}, LR:{5}'.format(\
                dkinit,dk,\
                un2str(uexp(B).nominal_value,uexp(B).std_dev(),2),\
                un2str(uexp(-OH0).nominal_value,uexp(-OH0).std_dev(),2),\
                un2str(uexp(-OH1).nominal_value,uexp(-OH1).std_dev(),2),\
                np.exp(LR)
                )
            #Note, INVERSE OCCAM FACTORS ARE PRINTED, MORE EASILY INTERPRETED, PERHAPS
        print '---------------------------------'
        for k,dk in enumerate(datalist):
            #comparison with 'previous' dataset
            if k==0: continue #(skip initial dataset since we are comparing to it
            
            dkprev = datalist[k-1]
            
            H0evA = dsetevs[Halt][dkprev][0]    #only difference from first loop is 'init' changed to 'prev'
            H0evB = dsetevs[Halt][dk][0]
            H0maxlogL = dsetevs[Halt][dk][1]
            H1evA = dsetevs[H0][dkprev][0]
            H1evB = dsetevs[H0][dk][0]
            H1maxlogL = dsetevs[H0][dk][1]
            
            B, LR, OH0, OH1 = computeBayesFact((H0evA,H0evB,H0maxlogL),(H1evA,H1evB,H1maxlogL)) #returns B, LR, OH0, OH1
            Bresults[Halt][H0]['{0}->{1}'.format(dkprev,dk)] = \
                computeBayesFact((H0evA,H0evB,H0maxlogL),(H1evA,H1evB,H1maxlogL)) #returns B, LR, OH0, OH1
            print '{0:<10}-->{1:<10} ; B:{2}, OH0:{3}, OH1:{4}, LR:{5}'.format(\
                dkprev,dk,\
                un2str(uexp(B).nominal_value,uexp(B).std_dev(),2),\
                un2str(uexp(-OH0).nominal_value,uexp(-OH0).std_dev(),2),\
                un2str(uexp(-OH1).nominal_value,uexp(-OH1).std_dev(),2),\
                np.exp(LR)
                )
    print '---------------------------------'

    
#Ok lets make some graphs to make it obvious for everybody

w = 1./5

colors = ['r','b','g','m']

def autolabel(rects,lbls):
    # attach some text labels
    for rect,lbl in zip(rects,lbls):
        height = rect.get_height()
        b = rect.get_y()
        print height, b
        s = height*0.8 #extra shift of label
        midpoint = np.sqrt((b+height)*b)      #geometric mean is just the standard mean in log space
        #if label is too close to the zero line, shift it downwards
        if midpoint < 1:    #only do this for downwards pointing bars
            if midpoint > 1./3. and height<3:
                midpoint = midpoint*0.7
            if b+height < 1 and b+height> 0.01:
                midpoint = midpoint*0.7
        plt.text(rect.get_x()+rect.get_width()/2., midpoint, lbl,
                ha='center', va='bottom', color='white', weight='bold')

def makelogbar(ax,fig,data,group_labels,alpha,legend=True):
    """Make a stacked bar graph
    Args:
    ax - subplot in which to create bars
    fig - figure object containing ax
    data - N*M array of bar heights; N columns of M stacked bars will be created.
    group_labels - list of N labels to give each stack of bars.
    alpha - list of M alpha values to give to each stack of bars, for visual variety.
    legend - include a legend in the output plot.
    """
    
    #print data
    #need to build up stacked bar graph. Is a bit convoluted so I give up trying to explain how it works.
    rects=[]    #store plot data
    lbls=[]
    Nscenarios=len(data)
    Ndatasets=len(data[0])
    print data
    print 'Nscenarios ', Nscenarios
    print 'Ndatasets ', Ndatasets
    x=np.arange(Nscenarios+1)
    print x
    for i,row in enumerate(data):
        #if i==0: continue #skip the first scenario, this is the SM stuff
        print row
        b=1   #set initial "bottom" position for bars (total evidence, or cumulative bayes factor
        #stack bars for this scenario
        left=x[i]  #left position for the set of bars
        for j in range(Ndatasets): 
            #if j==1 or j==2: continue
            y = row[j].nominal_value
            dy = row[j].std_dev()
            print j, y, dy
            newb = b*y #multiply latest bayes factor in to total
            h = b - newb
            b = newb
            print "Scenario", i, "dataset", j, "bar", j + Ndatasets*i 
            rects += ax.bar(left, h, 1, b, color=colors[j], alpha=alpha[i], log=True)
            print 'bar height', y, 1/y
            if y<1:
                if int(1./y+0.5)==1 :
                    print 'NO LABEL', 1./y, int(1./y+0.5)
                    lbls += ['']
                elif 1./y>=100:
                    lbls += ['{0}'.format(int(np.round(1/y)))]
                else:
                    lbls += ['{0}'.format(np.round(1/y,decimals=1))]
            else:
                if int(y+0.5)==1 :
                    print 'NO LABEL', y, int(y+0.5)
                    lbls += ['']
                elif y>=100:
                    lbls += ['{0}'.format(int(np.round(y)))]
                else:
                    lbls += ['{0}'.format(np.round(y,decimals=1))]
        #Stacked bar built, now tag it with the height of the bar
        y=b
        if y<1:
            if int(1./y+0.5)==1 :
                print 'NO LABEL', 1./y, int(1./y+0.5)
                lbl = ''
            elif 1./y>=1000:
                lbl = '{0}'.format(int(np.round(.1/y)*10))
            elif 1./y>=100:
                lbl = '{0}'.format(int(np.round(1/y)))
            else:
                lbl = '{0}'.format(np.round(1/y,decimals=1))
            pos = y/4
        else:
            if int(y+0.5)==1 :
                print 'NO LABEL', y, int(y+0.5)
                lbl = ''
            elif y>=100:
                lbl = '{0}'.format(int(np.round(y)))
            else:
                lbl = '{0}'.format(np.round(y,decimals=1))
            pos = y*2
        if float(lbl)>1e4:
            lbl = '{0}e4'.format(np.round(float(lbl)/1.e4,decimals=1))
        plt.text(
            rects[-1].get_x()+rects[-1].get_width()/2., pos, lbl,
            ha='center', va='bottom', color='black', weight='bold')
        
    #Draw a big fat line over the axis to emphasis the 'equal odds' line
    xeq = np.arange(0,x[Nscenarios]+1,1)
    yeq = xeq*0+1
    ax.plot(xeq,yeq, lw=3, color='k')

    #format bar chart
    
    # This sets the ticks on the x axis to be exactly where we put
    # the center of the bars.
    ax.set_xticks(x+0.5)
    # Set the x tick labels to the group_labels defined above.
    ax.set_xticklabels(group_labels)
    ax.set_xlim(0,x[Nscenarios])
    """
    yticks=[1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5]
    ylabs=['$1:10^5$','$1:10^4$','$1:10^3$','$1:100$','$1:10$','$1:1$','$10:1$','$100:1$','$10^3:1$','$10^4:1$','$10^5:1$']
    ax.set_yticks(yticks)
    #ax.set_yticks(minor=True)
    ax.set_yticklabels(ylabs)
     
    # Labels for the ticks on the x axis.  It needs to be the same length
    # as y (one label for each bar)
    ax.set_ylim(1e-5,1e3)    
    """
    # Extremely nice function to auto-rotate the x axis labels.
    # It was made for dates (hence the name) but it works
    # for any long x tick labels
    fig.autofmt_xdate()
    
    if legend:
        leg = plt.legend((rects[0], rects[1], rects[2]), ('LEP+Xenon', 'SUSY search', 'Higgs search'), 1)
        leg.get_frame().set_alpha(0.3)
    
        ltext  = leg.get_texts()  # all the text.Text instance in the legend
        plt.setp(ltext, fontsize='small')

    #print lbls
    autolabel(rects,lbls)
    
    return 
    
#Need to add in likelihood ratio contributions to evidence. Only the higgs
#search maximum likelihood values are different in the CMSSM and SM so we will just
#add this one case in manually.

#need to go through each scenario and compute the likelihood ratios for each piece of data
"""
higgsmaxlogLratio = np.exp(np.max(LRset[:,3])) #(L_CMSSM / L_SM)



#slight hack for now: just take average
print 'higgsmaxlogLratio', higgsmaxlogLratio
LRset = np.array([[1,1,higgsmaxlogLratio]]) #likelihood ratios for each data change. Should be (L_CMSSM / L_SM) I think.
print 'likelihood ratios:', LRset
print 'occam factors:', occam
data1 = np.append(LRset,occam,axis=0)
ddata1 = np.append([[0,0,0]],doccam,axis=0)
#print 'appended:', data1
"""

# B, LR, OH0, OH1
# 0, 1, 2, 3

HSM = altscenarios[0]  #SM
Hlog = 'log'  #SM Occam factors are the same for all CMSSM scenarios so just pick one

LRsall = [[(np.exp(-Bresults[HSM][H0][dk][1]),np.exp(-Bresults[HSM][H0][dk][1])*0.01)
 for dk in datachanges[1]] for H0 in scenarios]    #negative sign added to flip the ratio, I want SM being favoured to point downwards.

LRs = [[un.ufloat([max([x[0] for x in column(LRsall,i)]), np.std([x[0] for x in column(LRsall,i)])]) \
        for i in range(len(LRsall[0]))]] #take max of the ratio over all scenarios (as they should be the same in all scenarios)
# /np.max([x[0] for x in column(LRsall,i)])

SMoccams = [[uexp(-Bresults[HSM][Hlog][dk][2]) for dk in datachanges[1]]]    #negative sign added to flip the ratio (SM in the denominator)

CMSSMoccams = [[uexp(Bresults[HSM][H0][dk][3]) for dk in datachanges[1]] for H0 in scenarios]    
CMSSMlog10occams = [[Bresults[HSM][H0][dk][3]*np.log10(np.exp(1)) for dk in datachanges[1]] for H0 in scenarios]    
 
print CMSSMoccams

data1 = LRs + SMoccams + CMSSMoccams
print data1

fig= plt.figure(figsize=plotsize)
ax = fig.add_subplot(121)#,axisbg='0.9')

#shade regions in which LR and SM occam bars will be drawn
y=[1e-5,1e5]
x1=[0,0]
x2=[1,1]
x3=[2,2]
ax.fill_betweenx(y,x1,x2,color='y',alpha=0.2,lw=0)
ax.fill_betweenx(y,x2,x3,color='b',alpha=0.2,lw=0)

makelogbar(ax,fig,data1,\
    group_labels=['LR','SM','log','CCR','log, no $\delta a_\mu$','CCR, no $\delta a_\mu$'],\
    alpha=[.7,.7,.7,.7,.7,.7,.7],legend=False)#alpha=[.7,.5,.9,.7,.5,.3],
    
plt.ylabel('SM $\longleftarrow$  $\longrightarrow$ CMSSM')
ax.set_title('Bayes factor contributions')
ax.yaxis.grid(True)#, which='minor') 
yticks=[1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5]
ylabs=['$1:10^5$','$1:10^4$','$1:10^3$','$1:100$','$1:10$','$1:1$','$10:1$','$100:1$','$10^3:1$','$10^4:1$','$10^5:1$']
ax.set_yticks(yticks)
ax.set_yticklabels(ylabs)
ax.set_ylim(1e-5,1e3)    


#draw some error bars on an overlayed axis (to avoid the log scaling)
""" TO SMALL TO BOTHER WITH
ax2 = fig.add_subplot(121, frameon=False)

xerrs = [2.5,3.5,4.5,5.5]
yerrs = [val.nominal_value for val in [np.sum(row) for row in CMSSMlog10occams]] 
print 'yerrs', yerrs
errs = [val.std_dev() for val in [np.sum(row) for row in CMSSMlog10occams]]
print 'errs', errs
ax2.errorbar(xerrs, yerrs, yerr=errs, fmt='.', ecolor='k', elinewidth=3)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlim(0,6)
ax2.set_ylim(-5,3)   
"""

#======================
#NOW DO IT AGAIN, BUT WITH THE SM EVIDENCES DIVIDED IN DIRECTLY

#add together LR and occam pieces of the Bayes factor
data1 = np.array(data1)

data2 = np.array([data1[i] * data1[0] * data1[1] for i in range(2,len(data1))])
#data1[0] - likelihood ratio contribution to Bayes factor. Should be (L_CMSSM / L_SM) I think.
#data1[1] - SM Occam factor contribtuion to Bayes factor
print 'data2', data2

data2 = [[uexp(-Bresults[Halt][H0][dk][0]) for dk in datachanges[1]] for Halt in altscenarios for H0 in scenarios]    #negative sign added to flip the ratio, I want SM being favoured to point downwards.

print data2

ax = fig.add_subplot(122)#,axisbg='0.9')

makelogbar(ax,fig,data2,\
    group_labels=['log','CCR','log, no $\delta a_\mu$','CCR, no $\delta a_\mu$'],\
    alpha=[.7,.7,.7,.7])#alpha=[.9,.7,.5,.3]
ax.set_title('Bayes factors (B)')
ax.yaxis.grid(True)#, which='minor') 
ax.set_ylim(1e-5,1e3) 
ax.yaxis.set_label_position("right")
ax.set_ylabel("$10\log_{10}B$ (decibans)")
yticks=[1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5]
ylabs=['$-50$','$-40$','$-30$','$-20$','$-10$','$0$','$10$','$20$','$30$','$40$','$50$']
ax.set_yticks(yticks)
ax.set_yticklabels(ylabs)
ax.set_ylim(1e-5,1e3)   
#adjust spacing of subplots
fig.subplots_adjust(wspace=0.1)
ax.yaxis.tick_right()
ax.yaxis.set_ticks_position('both')

#p1 = plt.bar(ind, menMeans,   width, color='r', yerr=womenStd)
#p2 = plt.bar(ind, womenMeans, width, color='y',
#             bottom=menMeans, yerr=menStd)

#ax.set_yscale('log')

#ax.bar(x, data[0], w, color='r')

filename = 'BayesFactorContributions'
fig.savefig("{0}/{1}.pdf".format(outdir,filename),dpi=(800/8))

    
    
    

quit()

#old stuff
    
#we will compare all evidences to the first and the one previous in each list.
scenarios=[datalistSM,datalist1,datalist2,datalist3,datalist4]

occam=np.ones(np.array(scenarios).shape)
doccam=np.ones(np.array(scenarios).shape)
print occam
for j,datalist in enumerate(scenarios):
    evs=[]
    devs=[]
    Dev0A=[]
    dDev0A=[]
    Dev0B=[]
    dDev0B=[]
    maxlogL=[]
    for dk in datalist:
        evs+=[dsetevs[dk][0][0]]
        devs+=[dsetevs[dk][0][1]]
        maxlogL+=[dsetevs[dk][1]]
    #compare to 'initial' evidence
    #!!!!!!!!!!!!!!!!!!1fix up error computation!!!!!!!!!!!!!!!!!!!!
    print '---------Evidence changes--------'
    for i in range(1,len(evs)):
        eDev0, deDev0, dinveDev0 = DlogevstoDev((evs[0],devs[0]),(evs[i],devs[i]))
        Dev0A+=[eDev0]
        dDev0A+=[deDev0]
        print '{0:<10}-->{1:<10} : {2} or ~1/{3} ; i.e. 1:1-->1:{4:.3g}'.format(\
            datalist[0],datalist[i],un2str(eDev0,deDev0,2),\
            un2str(1/eDev0,dinveDev0,2),1/eDev0)
    print '---------Bayes factors-----------'
    #Assumes alternate model is first scenario
    for i in range(1,len(Dev0A)):
        print '{0:<10}-->{1:<10} : {2} or ~1/{3} ; i.e. 1:1-->1:{4:.3g}'.format(\
            datalist[0],datalist[i],un2str(Dev0A[i]/Dev0A[0],dDev0A[i],2),\
            un2str(1/(Dev0A[i]/Dev0A[0]),dinveDev0,2),1/(Dev0A[i]/Dev0A[0]))
    print '--------Evidence changes-----------'
    #compare to 'previous' evidence
    for i in range(1,len(evs)):
        eDev0, deDev0, dinveDev0 = DlogevstoDev((evs[i-1],devs[i-1]),(evs[i],devs[i]))
        #compute occam factors for use in bar charts
        #print 'dataset:', datalist[i]
        #print 'Dev:', eDev0
        #print 'maxLogL:', maxlogL[i]
        #print 'maxL:', np.exp(maxlogL[i])
        #print 'occam:', eDev0 / np.exp(maxlogL[i])
        occam[j,i]=eDev0 / np.exp(maxlogL[i])
        doccam[j,i]=deDev0 #modify by uncert in likelihood?
        inveDev0 = 1/eDev0
        try:
            strval = un2str(inveDev0,dinveDev0,2)
        except OverflowError:   #handle Dev=0 case
            strval = '1'
        print '{0:<10}-->{1:<10} : {2} or ~1/{3} ; i.e. 1:1-->1:{4:.3g}'.format(\
           datalist[i-1],datalist[i],un2str(eDev0,deDev0,2),\
            strval,inveDev0)
    print '---------------------------------'
    print '--------Evidence changes-----------'
    #compare to 'previous' evidence
    for i in range(1,len(evs)):
        eDev0, deDev0, dinveDev0 = DlogevstoDev((evs[i-1],devs[i-1]),(evs[i],devs[i]))
        #compute occam factors for use in bar charts
        #print 'dataset:', datalist[i]
        #print 'Dev:', eDev0
        #print 'maxLogL:', maxlogL[i]
        #print 'maxL:', np.exp(maxlogL[i])
        #print 'occam:', eDev0 / np.exp(maxlogL[i])
        occam[j,i]=eDev0 / np.exp(maxlogL[i])
        doccam[j,i]=deDev0 #modify by uncert in likelihood?
        inveDev0 = 1/eDev0
        try:
            strval = un2str(inveDev0,dinveDev0,2)
        except OverflowError:   #handle Dev=0 case
            strval = '1'
        print '{0:<10}-->{1:<10} : {2} or ~1/{3} ; i.e. 1:1-->1:{4:.3g}'.format(\
           datalist[i-1],datalist[i],un2str(eDev0,deDev0,2),\
            strval,inveDev0)
    print '---------------------------------'


