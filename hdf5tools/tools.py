
"""PySUSY CMSSM evidence project hdf5 filesystem creation

This is a library of tools (mostly functions) designed for storing and
accessing PySUSY datasets in a hdf5 filesystem"""

import numpy as np
import os
import array
import re
import operator
import shlex

from hdf5tools.import_tools import * #my hdf5 import tools

from StringIO import StringIO

import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib as mpl

import scipy.stats as sps

from itertools import groupby
from operator import itemgetter

import time 

#import uncertainties as un
#from uncertainties.umath import exp as uexp

#global options that may be desired to be tweaked
font = {'family' : 'serif',
        'weight' : 'normal',  #'bold'
        'size'   : 16,
        }
mpl.rc('font', **font)
#mpl.rc('text', usetex=True)

       
#==============ANALYSIS FUNCTIONS=======================================

#custom colormap for Delta chi^2 plots

#Define contours of likelihood function to plot (in sigma units)
#contsigmas = [0.5,1,2,3,4,5,6,7]
#contsigmas = [1,2]    #reduced set to remove clutter
#rellevels = np.array(contsigmas)**2 #square every element to get them into chi^2 units

#compute delta chi^2 for 68 and 95% 2D confidence regions 
CLs = [0.68,0.95]
df = 2  #two degrees of freedom in 2D profile likelihood plot
rellevels = [sps.chi2.isf(1-CL,df) for CL in CLs]   #isf = inverse survival function (inverse of 1-cdf). returns chi2 value at boundary of specified confidence region

#Define colormap for chi^2 plots
#scaled colors go from 0 to 1, so we need to set a max value (chi^2=25 sounds good)
#actually plot "sigmas", i.e. sqrt(chi^2), so max is then 5.
mn=0
mx=5
s0=0./(mx-mn)
s1=1./(mx-mn)
s2=2./(mx-mn)
s3=3./(mx-mn)
s4=4./(mx-mn)
s5=5./(mx-mn)

#colors are in order ('green':(0., 1., 0.), 'yellow': (1., 1., 0.), 'orange' : (0., 0.5, 1.)
# 'red' : (0., 0., 1.), 'darkred' : (0., 0., 0.5), 'darkerred' : (0., 0., 0.2) (names just made up)
#note, the rgb values are a bit screwed up in the previous list, but they are correct in the cdict
cdict = {
'red'  :  ((s0, 0., 0.), (s1, 1., 1.), (s2, 1., 1.), (s3, 1., 1.), (s4, .5, .5), (s5, .2, .2)),
'green':  ((s0, 1., 1.), (s1, 1., 1.), (s2, .5, .5), (s3, 0., 0.), (s4, 0., 0.), (s5, 0., 0.)),
'blue' :  ((s0, 0., 0.), (s1, 0., 0.), (s2, 0., 0.), (s3, 0., 0.), (s4, 0., 0.), (s5, 0., 0.))
}

#generate the colormap with 1024 interpolated values
chi2cmap = colors.LinearSegmentedColormap('chi2_colormap', cdict, 1024)
chi2cmap.set_bad('w',1.)    #set color (and alpha) for bad ('masked') values

#cutsom colormap for marginalised posterior plots

cdict2 = {
'red'  :  ((0., 1., 1.), (1., 0., 0.)),
'green':  ((0., 1., 1.), (1., 0., 0.)),
'blue' :  ((0., 1., 1.), (1., 0., 0.))
}

cdict3 = {
'red'  :  ((0., 0., 0.), (1., 1., 1.)),
'green':  ((0., 0., 0.), (1., 1., 1.)),
'blue' :  ((0., 0., 0.), (1., 1., 1.))
}

cdict4 = {
'red'  :  ((0., 1., 1.), (.01, 0., 0.), (.05, .1, .1), (1., .9, .9)),
'green':  ((0., 1., 1.), (.01, .3, .3), (.05, .1, .1), (1., .9, .9)),
'blue' :  ((0., 1., 1.), (.01, .3, .3), (.05, .1, .1), (1., 1., 1.))
}

#generate the colormap with 1024 interpolated values
margcmap = colors.LinearSegmentedColormap('marg_colormap', cdict4, 1024)
margconts = [0.68,0.95]

def getcols(structarr,colnames,cleannans=False):
    """"helper function to retrieve a normal, unstructured numpy array from the
    structured array that the dataset returns
    Args
    structarr - result of dset[:], or dset[0:1000] if a subset of rows is desired
    colnames  - list of field names to retrieve from array
    cleannans - switch to turn on automatic removal of records containing nans.
    """
    """ This way may get the columns switched around!
    #As a bonus, remove NaN's from data
    
    data = structarr[:][colnames].view(dtype='<f4').reshape(-1,len(colnames))
    dataout = data[~np.isnan(data).any(1)]  #deletes rows containing nans
    print 'NaN count: ',len(data) - len(dataout)
    """
    #Pull columns out one at a time to ensure we get the correct order
    data = np.array([structarr[col] for col in colnames]).transpose()
    print data.shape, type(data)
    if cleannans:
        try:
            dataout = data[~np.isnan(data).any(1)]  #deletes rows containing nans
        except TypeError as err:
            print 'TypeError encountered, dumping extra data...'
            for i,row in enumerate(data):
                try:
                    ~np.isnan(row).any(1)
                except (TypeError,NotImplementedError) as err2:
                    print 'TypeError or NotImplementedError occurred running \
isnan function on row {0}, dumping row...'.format(i)
                    print row, type(row)
                    raise err2
        print dataout.shape
        print 'NaN count: ',len(data) - len(dataout)
    else:
        dataout = data
    return dataout
    
def listgetcolsT(structarr,listoflists):
    """A wrapper for getcols that facilitates use of several parameters lists
    at once. Note that each element of output is TRANSPOSED relative to
    standard getcols output"""
    #Join parameter lists into one big list
    flatlist = []
    for listi in listoflists:
        flatlist+=listi
    #get those columns from dataset
    flatout = getcols(structarr,flatlist).T
    #reshape flatout to match input listoflists and return it
    shapedout = []
    i=0
    for listi in listoflists:
        iend = i+len(listi)
        add = flatout[i:iend]
        shapedout += [ add ]
        i = iend
    return shapedout

def getcolsasstruct(structarr,colnames,mask=None):
    """Does the same as getcols, but retains the structured array format
    Essentially it's main purpose is to clean out any nan's
    Args
    structarr - result of dset[:], or dset[0:1000] if a subset of rows is desired
    colnames  - list of field names to retrieve from array
    mask - optional boolean mask; use to remove desired rows (i.e. rows from
        errornous points) 
    """
    if mask!=None:
        data = structarr[colnames][mask]
    else:
        data = structarr[colnames]
    print data.shape, type(data)
    
    # NEED TO ADD SOME ERROR CHECKING
    
    # Need to check for nan's field by field, then merge results
    mask2 = np.ones(len(data))
    for field,dt in data.dtype.descr:
        # combine mask of nan points from this field with rest of mask
        mask2 = np.logical_and(mask2, ~np.isnan(data[field]))
    
    # use mask to select only rows without any nans
    dataout = data[mask2]    
    print dataout.shape
    print 'NaN count: ',len(data) - len(dataout)
    return dataout

def bin2d(data,nxbins,nybins,binop='min',doconts=False):
    """2d binning algorithm
    Args
    data - n rows * 3 column dataset, columns in order x,y,z. Binning done
    by x,y
    binop - Operation to perform on bins. Available options:
        'min' - Returns the minimum z value for each bin
        'max' - Returns the maximum z value for each bin
    nxbins - Number of bins to use in x direction
    nybins -    "      "              y   "
    """
    x = data[:,0]
    y = data[:,1]
    #xbins = np.arange(min(x),
    zvals = data[:,2]
    eps = 1e-10     #shift for bin edge computation

    xindexes = np.floor((1-eps)*(nxbins-1)*(x-min(x))/(max(x)-min(x)))
    yindexes = np.floor((1-eps)*(nybins-1)*(y-min(y))/(max(y)-min(y)))
    #print xindexes
    #print yindexes
    
    outarray = np.zeros((nxbins,nybins))
    #print outarray
    
    #grouped=[list(value) for key, value in groupby(sorted(zip(xindexes,yindexes,zvals)), key=itemgetter(0,1))]
    grouped=[list(value) for key, value in groupby(sorted(zip(xindexes,yindexes,zvals)), key=itemgetter(0,1))]
    bininds=[nybins*bin[0][0] + bin[0][1] for bin in grouped]  #these are the FLAT indices in the target array

    if binop=='min':
        binvals=[min([el[2] for el in bin]) for bin in grouped]
        #outarray[:,:] = 1e300
        outarray[:,:] = np.nan
    elif binop=='max':
        binvals=[max([el[2] for el in bin]) for bin in grouped]
        outarray[:,:] = 0
    elif binop=='sum':
        binvals=[sum([el[2] for el in bin]) for bin in grouped]
        outarray[:,:] = 0
    else:
        raise ValueError('Invalid binop value ({0}) supplied to bin2d'.format(binop))

    #print bininds
    try:
        outarray.flat[bininds] = binvals
    except ValueError:
        print "WARNING: nan's present in bin index lists, input data may\
be degenerate in one or more dimensions."
        raise  
    #Recompute the bins so that each one contains the total probability enclosed by the
    #iso-probability-density contour on which it sits (use this to compute smallest 68%, 95% 
    #Bayesian credible regions) (DESIGNED FOR USE WHEN BIN VALUE IS A PROBABILITY MASS)
    if binop=='sum' and doconts==True:
        outarray2 = np.zeros((nxbins,nybins))
        outarray2[:,:] = 1.
        sb = np.array(sorted(zip(bininds,binvals), key=itemgetter(1), reverse=True)) #sort bins by probability
        #print zip(sb[:,1],np.cumsum(sb[:,1]))[0:20]
        #print zip(sb[:,1],np.cumsum(sb[:,1]))[-20:-1]
        outarray2.flat[list(sb[:,0])] = list(np.cumsum(sb[:,1]))         #cumulative sum of probabilities
        return outarray.transpose(), outarray2.transpose()
        
    return outarray.transpose()
    
def bin1d(data,nbins,binop='min'):
    """1d version of bin2d (in fact uses bin2d directly)
    Args:
    data - n rows * 2 column dataset, columns in order x,z. Binning done
    in x, z being the "height" or "density" to be binned.
    binop - Operation to perform on bins. Available options:
        'min' - Returns the minimum z value for each bin
        'max' - Returns the maximum z value for each bin
    nbins - Number of bins to use 
    """
    x = data[:,0]
    y = np.zeros(x.shape[0]) #"fake" y values to emulate the 2D binning
    y[:x.shape[0]/2] = 1  #need at least two different y values so ymax - ymin != 0.
    z = data[:,1]
    return bin2d(np.array(zip(x,y,z)),nbins,1,binop,doconts=False)[0]      #return 1d array of x vs binned values

def bin1dxy(data,nbins,binop='min'):
    """wrapper for bin1d that returns an array of x,y values, not just the
    binned y values
    """
    x = data[:,0]
    dx = (max(x)-min(x))/nbins
    xvals = np.arange(min(x),max(x)-dx/2.,dx)
    ybinned = bin1d(data,nbins,binop='min')
    print xvals.shape, ybinned.shape
    return np.vstack((xvals,ybinned)).T
    
def forceAspect(ax,aspect=1):
    extent = ax.get_window_extent().get_points()
    print extent
    ax.set_aspect(abs((extent[0,1]-extent[0,0])/(extent[1,1]-extent[1,0]))/aspect)
    
    """
    print ax
    print ax.get_images()
    if isinstance(ax, axes.Axes):
        extent =  ax.get_extent()
    elif isinstance(ax.get_images()[0], plt.axes.Axes):
        im = ax.get_images()
        extent =  im[0].get_extent()
    else:
        raise TypeError("Supplied 'ax' was not an Axes object or list of Axes objects!")
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
    """
    
#--------------main plotting routines-----------------------------------

def chi2scatplot(ax,data,title=None,labels=None,alpha=1.,overlay=False):
    """Creates a scatter plot of the data, colored by Delta chi^2 value
    Args:
    ax - Axis object on which to create plot
    data - 3 column numpy array with
        data[:,0] - x data
        data[:,1] - y data
        data[:,2] - chi^2 data
    alpha - set alpha level
    noaxes - turn off all axes and decorations (useful for overlays)
    """
    data = data[data[:,2].argsort()[::-1]] #sort points by chi2 (want to plot lowest chi2 points last, achieved by reversing sorted indices via '::-1')
    if overlay==True: lims = ax.axis()
    plot = ax.scatter(data[:,0],data[:,1],c=np.sqrt(data[:,2]-min(data[:,2])),s=1,lw=0,cmap=chi2cmap, norm=colors.Normalize(vmin=mn,vmax=mx,clip=True),alpha=alpha)
    if overlay==False:
       # Skip all this stuff if we are just drawing on extra data
       if title: ax.set_title(title)
       if labels: ax.set_xlabel(labels[0])
       if labels: ax.set_ylabel(labels[1])
       ax.set_xlim(min(data[:,0]),max(data[:,0]))
       ax.set_ylim(min(data[:,1]),max(data[:,1]))
       ax.grid(True)
    else:
       ax.axis(lims) #make sure the old limits weren't messed up

    return plot

def chi2logscatplot(ax,data,title=None,labels=None,logaxis='y'):
    """Creates a scatter plot of the data, colored by Delta chi^2 value
    """
    data = data[data[:,2].argsort()[::-1]] #sort points by chi2 (want to plot lowest chi2 points last, achieved by reversing sorted indices via '::-1')
    plot = ax.scatter(data[:,0],data[:,1],c=np.sqrt(data[:,2]-min(data[:,2])),s=1,lw=0,cmap=chi2cmap, norm=colors.Normalize(vmin=mn,vmax=mx,clip=True))
    if 'x' in logaxis: ax.set_xscale('log')
    if 'y' in logaxis: ax.set_yscale('log')
    if title: ax.set_title(title)
    if labels: ax.set_xlabel(labels[0])
    if labels: ax.set_ylabel(labels[1])
    ax.set_xlim(min(data[:,0]),max(data[:,0]))
    ax.set_ylim(min(data[:,1]),max(data[:,1]))
    ax.grid(True)
    return plot
    
def profplot(ax,data,title=None,labels=None,nxbins=np.floor(1.618*100),nybins=100):
    """Creates a binned, profiled plot of the data, colored by Delta chi^2 value,
    i.e. profile likelihood.
    """
    x = data[:,0]
    y = data[:,1]
    wx= (max(x)-min(x))/nxbins
    wy= (max(y)-min(y))/nybins
    print min(x),max(x),wx
    print min(y),max(y),wy
    xlist = np.arange(min(x),max(x)-wx*10e-3,wx)   #tiny shift to prevent nbins+1 bins in X and Y arrays
    ylist = np.arange(min(y),max(y)-wy*10e-3,wy)
    print len(xlist), len(ylist)
    
    outarray = bin2d(data,nxbins,nybins,binop='min')
    #print outarray[0:20,0:20]
    X, Y = np.meshgrid(xlist,ylist)
    minchi2 = np.nanmin(outarray)  #get min ignoring nans
    #print X.shape, Y.shape, outarray.shape, minchi2
    
    #Set special color for NaN values (see definition of colormap also)
    Dchi2 = np.sqrt(outarray - minchi2)
    #print Dchi2[0:20,0:20]

    masked_array = np.ma.array(Dchi2, mask=np.isnan(Dchi2)) #mask out the nan's; the masked values will be given a special color in the output plot
    
    #print masked_array[0:20,0:20]
    
    im = ax.imshow(masked_array, origin='lower', interpolation='nearest',
                    extent=(min(xlist),max(xlist)+wx,min(ylist),max(ylist)+wy),
                    cmap=chi2cmap, norm=colors.Normalize(vmin=mn,vmax=mx,clip=True), aspect='auto')
    
    CS = ax.contour(X+wx/2, Y+wy/2, outarray - minchi2, levels=rellevels, lw=3) #colors=['g','y','r']
    bfidx=np.where(outarray.flat==minchi2) #get flat index of best fit point
    ax.plot([X.flat[bfidx]], [Y.flat[bfidx]], "ko")
    #ax.clabel(CS, inline=1, fontsize=10)
    
    if labels: ax.set_xlabel(labels[0])
    if labels: ax.set_ylabel(labels[1])
    ax.set_xlim(min(data[:,0]),max(data[:,0]))
    ax.set_ylim(min(data[:,1]),max(data[:,1]))
    ax.grid(True)
    
    return im
    
def margplot(ax,data,title=None,labels=None,nxbins=np.floor(1.618*100),nybins=100):
    """Creates a binned marginalised plot of the data, colored by marginalised posterior
    density.
    """
    x = data[:,0]
    y = data[:,1]
    wx= (max(x)-min(x))/nxbins
    wy= (max(y)-min(y))/nybins
    xlist = np.arange(min(x),max(x)-wx*10e-3,wx)    #tiny shift to prevent nbins+1 bins in X and Y arrays
    ylist = np.arange(min(y),max(y)-wy*10e-3,wy)
    
    outarraydens, outarrayconts = bin2d(data,nxbins,nybins,binop='sum',doconts=True)
    X, Y = np.meshgrid(xlist,ylist)
    maxpoint = max(outarraydens.flat)

    print X.shape, Y.shape, outarraydens.shape, outarrayconts.shape, max(outarrayconts.flat)
    
    im = ax.imshow(outarraydens, origin='lower', interpolation='nearest',
                    extent=(min(xlist),max(xlist)+wx,min(ylist),max(ylist)+wy),
                    cmap=margcmap, aspect='auto')
    
    CS = ax.contour(X+wx/2, Y+wy/2, outarrayconts, levels=margconts, lw=3, colors=[(0,1,0),(1,1,0)])    #(r,g,b)
    bfidx=np.where(outarraydens.flatten()==maxpoint) #get flat index of best fit point
    ax.plot([X.flat[bfidx]], [Y.flat[bfidx]], "ko")
    #ax.clabel(CS, inline=1, fontsize=10)
    
    if labels: ax.set_xlabel(labels[0])
    if labels: ax.set_ylabel(labels[1])
    ax.set_xlim(min(data[:,0]),max(data[:,0]))
    ax.set_ylim(min(data[:,1]),max(data[:,1]))
    ax.grid(True)
    
    return im

def margplot1D(ax,data,title=None,labels=None,trim=True):
    """Creates a 1D binned marginalised plot of the data
    Args:
    data[:,0] - x data
    data[:,1] - corresponding posterior masses 
    trim - Cut off outer bins containing less than 0.01% probability mass
    """
    nxbins=200

    x = data[:,0]
    wx= (max(x)-min(x))/nxbins
    xlist = np.arange(min(x),max(x)-wx*10e-3,wx)    #tiny shift to prevent nbins+1 bins in X and Y arrays
    
    outarraydens = bin1d(data,nxbins,binop='sum')
    
    # Perform trimming:
    # Do binning with all the data, cut off bins that contain almost no 
    # probability, then repeat binning over reduced range (iterate)
    if trim:
        while True:
            # from low
            posl = 0
            pmass = outarraydens[posl]
            while pmass<0.001:
                posl += 1
                pmass = outarraydens[posl]
            # from high
            posh = len(outarraydens)-1
            pmass = outarraydens[posh]
            while pmass<0.001:
                posh -= 1
                pmass = outarraydens[posh]
            # compute new range and repeat binning
            data = data[ np.logical_and(data[:,0]>xlist[posl], data[:,0]<xlist[posh+1]) ] 
            x = data[:,0]
            wx= (max(x)-min(x))/nxbins
            xlist = np.arange(min(x),max(x)-wx*10e-3,wx)    # tiny shift to prevent nbins+1 bins in X and Y arrays
            if posh-posl > 80:  # if we aren't cutting off many bins don't bother to optimise further
                break
            # repeat binning
            outarraydens = bin1d(data,nxbins,binop='sum')

    #im = ax.plot(xlist,outarraydens)   
    im = ax.bar(xlist,outarraydens,wx,linewidth=0,color='b',alpha=0.4)
      
    if labels: ax.set_xlabel(labels)
    ax.set_ylabel("Binned posterior mass")
    ax.set_xlim(min(data[:,0]),max(data[:,0]))
    
    return im
    """
    X, Y = np.meshgrid(xlist,ylist)
    maxpoint = max(outarraydens.flat)

    print X.shape, Y.shape, outarraydens.shape, outarrayconts.shape, max(outarrayconts.flat)
    
    im = ax.imshow(outarraydens, origin='lower', interpolation='nearest',
                    extent=(min(xlist),max(xlist)+wx,min(ylist),max(ylist)+wy),
                    cmap=margcmap, aspect='auto')
    
    CS = ax.contour(X+wx/2, Y+wy/2, outarrayconts, levels=margconts, lw=3, colors=[(0,1,0),(1,1,0)])    #(r,g,b)
    bfidx=np.where(outarraydens.flatten()==maxpoint) #get flat index of best fit point
    ax.plot([X.flat[bfidx]], [Y.flat[bfidx]], "ko")
    #ax.clabel(CS, inline=1, fontsize=10)
    
    if labels: ax.set_xlabel(labels[0])
    if labels: ax.set_ylabel(labels[1])
    ax.set_xlim(min(data[:,0]),max(data[:,0]))
    ax.set_ylim(min(data[:,1]),max(data[:,1]))
    ax.grid(True)
    
    return im
    """

def make1Dbinplots(axlist,xparlist,xparnames,ctunall,tunall,compardict=None,
                        nbins=100, binop='min', ymincut=0.1, plotkwargs=None,
                        ylab=None):
    """Intended for creation of plots of fine-tuning vs some parameters,
    but can be used to make any plots of a series of binned 1D data against
    various parameters.
    Args:
    axlist - list of axes to attach plots. Must match len(xparlist)
    xparlist - list of x data to bin. must match len(axlist), and each
      element must match len(ctunall[i])
    xparnames - labels to use for x axes and storage dictionaries
    ctunall - list of y data vectors to be binned against xpar data
    tunall - names of y data, for labels and storage dictionaries.
    
    Output:
       plots attached to axes in axlist
       dictionary containing binned data used for plots, with keys from
       'tunall' list. 
       
    Optional:
    compardict - dictionary of binned data of the same format as output
      dictionary. Will be plotted anywhere it matches up with the input
      parameter names.
    """
    if plotkwargs==None: plotkwargs = {}
    outdict={}
    for ax,xpar,xparname in zip(axlist,xparlist,xparnames):
        print 'Generating {0} plots...'.format(xparname)
        outdict[xparname] = {}
        for c,lab in zip(ctunall,tunall):
            print lab, xpar.shape, c.shape
            minbinnedxy = bin1dxy(np.vstack((xpar,c)).T, nbins, binop)
            outdict[xparname][lab] = minbinnedxy
            #skip plot if tuning always low
            try:
                compdata = compardict[xparname][lab]
                if ( max(minbinnedxy[:,1]) < ymincut ) and \
                   ( max(compdata[:,1]) < ymincut ) :
                   print 'skipping {0}vs{1} plot...'.format(xparname,lab),max(compdata[:,1]),max(minbinnedxy[:,1]),ymincut
                   continue
                ax.plot(*compdata.T, label=lab+'_comp', alpha=0.5, **plotkwargs)
            except (TypeError,KeyError) as err:
                if max(minbinnedxy[:,1]) < ymincut: continue
            ax.plot(*minbinnedxy.T, label=lab, **plotkwargs)
        leg = ax.legend(fontsize='x-small',ncol=2)
        try:
            leg.get_frame().set_alpha(0.5)
        except AttributeError:
            pass
        ax.set_xlabel(xparname)
        if ylab!=None: ax.set_ylabel(ylab)
    return outdict
    
#-------------------timing plots----------------------------------------

def timehist(times,ax1,ax2=None,labels=None,colors=None):
    """create two histograms of timing information: the first being the
    timing pdf, and the second being the 'fraction of cpu time spent in 
    loops of this speed' pdf.
    Args:
    ax1 - axis on which to draw first histogram
    ax2 - "  "                  second " (defaults to same as first)
    labels - passed on to ax.bar function label keyword
    colors - " "                          color "
    """
    if ax2==None: ax2=ax1 
    if colors==None: colors=['blue','red']
    
    #get min and max times excluding zero and inf (may appear in errornous lines)
    m, M = min(times[np.nonzero(times)]), max(times[np.isfinite(times)])
    if m == M:  raise ValueError("Error creating timing histogram! min and max \
run time are identical ({0} seconds)! Highly unlikely this is valid data") 
    numbins=100.
    w=(M-m)/numbins
    if w<20: w=20   #set minimum bin width to this
    countrange = np.arange(m,M,w)
    try:
        n, bins = np.histogram(times, countrange, normed=True)              #create histogram, normalised to have heights equal to the pdf values (NOT bin probability masses; to get these multiply by w)
    except IndexError:  
        print "Error creating timing histogram! dumping time data used..."
        print times[-100:]                                                  #n = bin pdf values
        print countrange
        print "min time: ", m
        print "max time: ", M
        raise                                                               #bins = bin edge values
    
    #count up 99% of the probability and neglect bins outside this (shrinks histograms to interesting region)
    totp=0
    for i,p in enumerate(n):
        totp+=p*w
        if totp>=0.99:
            break
    #if i<10 then redo the binning over this new range
    if i<50:
        M = bins[i+10]
        numbins=100.
        w=(M-m)/numbins
        if w<20: w=20   #set minimum bin width to this
        countrange = np.arange(m,M,w)
        n, bins = np.histogram(times, countrange, normed=True)
        i=len(n)   #index of last bin

    i = i - 1   #move to left hand side of last bin
    bincenters = .5*(bins[1:i+1]+bins[0:i])
    totaltime = sum(n[:i]*bincenters)
    #leave out the first bin since we don't really care about the loops
    #in which an error occurred quickly and were skipped
    nmax = max(n[1:i])
    nfracmax = max(n[1:i]*bincenters[1:]/totaltime)
    h1 = ax1.bar(bins[1:i],n[1:i]/nmax,w,linewidth=0,color=colors[0],alpha=0.4,label=labels[0])
    h2 = ax2.bar(bins[1:i],(n[1:i]*bincenters[1:]/totaltime)/nfracmax,w,linewidth=0,color=colors[1],alpha=0.4,label=labels[1])
    ax1.set_ylim((0,1))
    ax2.set_ylim((0,1))

#=======================================================================
# ANALYSIS AUTOMATION TOOLS
#=======================================================================

#Here we specify a class to be used for generating results and plots. It
#is similar in usage to the 'LinkDatabaseForImport' class, but for analysis rather
#than data import, and links to a particular dataset.

#This class is very specific to my current analysis and should not be expected to work
#in general

class LinkDataSetForAnalysis():
    """Create a wrapper object to provide an interface to the hdf5 database"""
    #-------Variables set by  __init___--------
    dset = None    #attribute to store the h5py file object of the database (or structured array subset of it)
    outdir = None
    effprior = False    #flag specifying if CCR prior was used for this dataset
    likepar = None
    probpar = None
    goodmask = None   #boolean mask identifying points for which no errors occurred 
    
    likedata = None     #global -2*loglikelihood column (potentially reweighted)
    probdata = None     #posterior probability column
    
    #----------Parameters-------------
    plotsize = (8,4)
    gridplotsize = (20,20)
    
    #------Variables set by addlive-------
       
    def getcols(self,colnames,cleannans=False):
        """Wrapper for standard "getcols" function, just converted to a method 
        for the analysis object"""
        return getcols(self.dset,colnames,cleannans) 
    
    def listgetcolsT(self,listcolnames):
        """Wrapper for standard "listgetcolsT" function, just converted to a method 
        for the analysis object"""
        return listgetcolsT(self.dset,listcolnames) 
    
    def getcolsasstruct(self,colnames,mask=None):
        """Wrapper for standard "getcolsasstruct" function, just converted to a 
        method for the analysis object"""
        return getcolsasstruct(self.dset,colnames,mask)     
    
    def __init__(self,dsetIN,outdir,allparsIN,likepar=None,neg2logl=True,probpar=None,\
                    effprior=None,timing=False,limitsize=int(1e7),lims=None):
        """Initialise link to dataset, create output directory and check
        that requested data columns are present in dataset. If optional
        arguments are given some extra processing is done.
            Args:
        dsetIN - hdf5 dataset (or structured array) containing data to analyse
        outdir - Path to directory to store results
        allparsIN - list of fieldnames of columns to to be extracted from dsetIN
        likepar - fieldname of -2*loglikelihood column
        neg2logl - (True/False); if False, treats 'likepar' as log-likelihood rather than -2*log-likelihood
        probpar - fieldname of posterior column
        effprior - fieldname of column containing effective log-likelihood
        reweighting factor from effective prior.
        limitsize - largest number of records allowed, datasets larger than this
            will be trimmed, preserving the highest likelihood points
        lims - A list of cuts to make on the dataset, 
            e.g. lims = [('M0',(0,10000)), ('M12',(0,10000))] #tuple=(min,max)
        """
        #make a copy of the input fieldname list to avoid modifying the user's 
        #original list
        allpars = allparsIN[:]  
          
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print "Storing plots in {0}...".format(outdir)
        print "limitsize: ", limitsize
        try:
            limitsize = int(limitsize) 
        except ValueError:
            limitsize = int(float(limitsize))

        if timing:
            fieldnames = dsetIN.keys()
        else:
            fieldnames = dsetIN.dtype.names

        print fieldnames
        
        if likepar: allpars+=[likepar]
        if probpar: allpars+=[probpar]
        if effprior: allpars+=[effprior]
        if timing==True:
            allpars+=['looptime','samplertime','liketime']
            allpars+=[prog for prog in fieldnames if 'prog' in prog]
            if 'errors' in fieldnames:
                allpars += ['errors']
        allpars=list(set(allpars))  #remove duplicates
        #print 'timing:', timing
        #print 'allpars:', allpars
        
        #Import a subset of the dataset to speed things up
        t0 = time.time() 
        allparsVERIFIED = list(set([name for name in allpars if name in fieldnames])) #make sure parameters specified by 'allpars' exist in dataset
        #reassess the optional arguments in case the requested fields were not found
        likepar =  likepar if likepar in allparsVERIFIED else False
        probpar = probpar if probpar in allparsVERIFIED else False
        effprior = effprior if effprior in allparsVERIFIED else False
        
        nreq = len(allpars)
        nfound = len(allparsVERIFIED)
        if nfound != nreq:
            print "WARNING! Not all fieldnames requested could be found in dataset! Continuing with those found. \
            (# requested = {0}, # found = {1}, # missing = {2})".format(nreq, nfound, nreq - nfound)
            print "Fieldnames not found: {0}".format([name for name in allpars if name not in allparsVERIFIED])
        
        #Extract selected columns from hdf5 dataset 
        #Not sure why I thought timing datasets need different treatment...
        #Ok now I remember, they are actually stored in the hdf5 file differently
        #to txt: stored with each column as an individual dataset! Need to change
        #txt datasets to also work this way
        #-Have just learned that unicode field names cause problems for 
        #numpy/h5py, so I am adding a new function that mangles the field names 
        #to ascii (unicode2ascii).
        if timing:
            print 'somecol = {0}'.format(allparsVERIFIED[-1])
            somecol = dsetIN[allparsVERIFIED[0]] #unknown column, for length
            print 'length of verified data:', len(somecol)
            newdtype = [(unicode2ascii(key),dset.dtype) for key,dset in dsetIN.items() if key in allparsVERIFIED]       #get the dtypes for the chosen columns 
            print newdtype
            try:
                #create output structured array
                # Check if size is within allowed limit   
                if len(somecol)>limitsize:
                    print "Warning, database contains more records than allowed by \
limitsize parameter (size={0}, allowed={1}). Taking only {1} entries with highest likelihood in dataset. \
'limitsize' can be set larger in the arguments to LinkDataSetForAnalysis, \
but this may result in 'array is too big' errors from numpy, depending on the \
number of fields ('columns') in the dataset".format(len(somecol),limitsize)
                    #sort by likelihood, preserving original indices so we can
                    #use them to grab the appropriate rows from the dataset
                    #!!!!!Sorting is way too slow for large datasets
                    #print 'Sorting likelihood values...'
                    #sortedlikes = sorted(enumerate(likecol), \
                    #                key=operator.itemgetter(1))
                    #lowest chi2 values first in sorted list
                    #print 'Generating mask to exclude lowest likelihood values...'
                    #mask = sortedlikes[:limitsize][:,1] #just grab the list of indices
                    ##==============
                    # Ahh, this is a better way: UPDATE: ok was slow for large datasets, sorting now happens externally
                    likecol = dsetIN[likepar][:limitsize] #force creation of numpy array, for fast sorting
                    initmask = np.array([True]*len(likecol))
                    #initmask = likecol < np.sort(likecol)[limitsize] #mask=True for n (=limitsize) lowest chi2 values
                    ## Err the below needs bottleneck module which is apparently not standard.
                    # Aha, doing a partial sort should speed things up:
                    #datamask = bottleneck.argpartsort(likecol, limitsize) #should get the indices of the lowest n=limitsize chi2 entries
                    cut = int(limitsize)
                    #cut = len(somecol)
                    print 'Dataset size after initial reduction: ', np.sum(initmask)
                else:
                    initmask = np.array([True]*len(somecol)) #default mask, gets ALL elements of database
                    cut = len(somecol)
                self.cut = cut  #need this for histogram creation
                # If more cuts are asked for by the user then figure them out now
                #datamask = np.array([True]*len(somecol[-cut:]))
                datamask = initmask[:]
                if lims:
                    initwhere = np.nonzero(initmask)[0] #get indices of True positions
                    print initwhere
                    print "Computing requested dataset cuts..."
                    for field,(cutmin,cutmax) in lims:
                        print "Performing cut: {0} <= {1} <= {2}".format(cutmin,field,cutmax) 
                        try:
                            datacol = dsetIN[field][:cut][initwhere]
                            #add new cut constraints to the mask
                            newmask = (datacol>=cutmin) & (datacol<cutmax)
                            # Combine this new mask back into the mask on the original large dataset
                            setFalse = initwhere[~newmask]
                            datamask[setFalse] = False
      
                        except ValueError:
                            print "Error while computing cuts! Dumping extra output..."
                            print len(datacol)
                            print len(datacol>=cutmin)
                            print len(datacol<=cutmax)
                            print len(datamask)
                            print field, cutmin, cutmax 
                            raise                                               
                print 'Dataset size after extra cuts imposed: ',np.sum(datamask)
                print 'Creating numpy storage space for dataset...'
                #print likecol[-1000:]
                self.dset = np.zeros(somecol[:cut][datamask].shape,dtype=newdtype)
            except ValueError as err:
                if err.message=="array is too big.":
                    print "ERROR! Attempted to create a dataset too large for numpy to \
handle. Please exclude some fields from the dataset and try again"
                    print "fields requested:", newdtype
                    print "number of rows in dataset:", len(dsetIN[allparsVERIFIED[0]])
                raise            
        else:
            # cuts specfied by "cut" are ignored if not using timing file
            cut = None

            # If more cuts are asked for by the user then figure them out now
            somecol = dsetIN[allparsVERIFIED[0]] #some unknown field, for length
            datamask = np.array([True]*len(somecol))
            if lims:
                    print "Computing requested dataset cuts..."
                    for field,(cutmin,cutmax) in lims:
                        datacol = dsetIN[field][:]
                        #add new cut constraints to the mask
                        try:
                            datamask = np.logical_and(datamask, 
                                np.logical_and(datacol>=cutmin, datacol<=cutmax)
                                                         )      
                        except ValueError:
                            print "Error while computing cuts! Dumping extra \
output..."
                            print len(datacol)
                            print len(datacol>=cutmin)
                            print len(datacol<=cutmax)
                            print len(datamask)
                            print field, cutmin, cutmax 
                            raise   
            newdtype = [(unicode2ascii(key),dt[0]) for key,dt in dsetIN.dtype.fields.items() if key in allparsVERIFIED]       #get the dtypes for the chosen columns 
            self.dset = np.zeros(somecol[datamask].shape,dtype=newdtype)     #create output structured array

        for par in allparsVERIFIED:
            print "Extracting {0} column...".format(par)
            self.dset[par] = dsetIN[par][:cut][datamask]       #loop through hdf5 dataset and extract columns into output array 
            #self.dset[par] = dsetIN[par][:][datamask]       #loop through hdf5 dataset and extract columns into output array 

        #self.dset = dsetIN[*allparsVERIFIED]
        print "Time taken importing dataset subset: ", time.time() - t0
            
        #If CCR prior was used for this dataset, we need to remove the CCR prior
        #"effective' likelihood contribution from the global likelihood to recover
        #'true' global likelihood. This likelihood was attached to the 'BQ' observable,
        #so it is stored in the effprior column.
        t0 = time.time()
        if effprior:
            try:
                print 'effprior:', effprior
                print self.dset[effprior][-cut:][datamask]
                effchi2 = -2 * np.array(self.dset[effprior][-cut:][datamask])    
                self.effprior = effchi2.any() #Check if any values for this
                #column are non-zero. If they are, we used an effective prior for this dataset.
            except ValueError:
                #if no effprior value exists...
                effchi2 = 0
                self.effprior = False
        #UPDATE: I think the above "except" is useless, will never be encountered
        #since effprior has the default value None.
        else:
            effchi2 = 0
            self.effprior = False
                    
        print "wtf2: ", time.time() - t0
        
        #Get -2*likelihood column and posterior column
        t0 = time.time() 
       
        if likepar:
            if neg2logl==True:
               self.likedata = np.array(self.dset[likepar]) - effchi2
            else:
               # Assume likepar is log-likelihood, not -2*loglikelihood
               self.likedata = -2*np.array(self.dset[likepar]) - effchi2 
            self.likepar = likepar
        else:
            print "Warning! No -2*loglikelihood column specified! When generating plots this data will need to be supplied to the plotting routines, or else an error will occur"
        if probpar: 
            self.probdata = np.array(self.dset[probpar])
            self.probpar = probpar
        self.outdir = outdir
        
        # Create mask telling us which points had no errors occur
        if timing and 'errors' in fieldnames:
            self.goodmask = self.dset['errors'] == ''
            if np.sum(self.goodmask)==0:
                print '"errors" dataset appears to mark every point as bad. Will assume that this is not correct (perhaps import not done quite correctly) and ignore this dataset'
                self.goodmask = None   
 
    def applygoodmask(self):
        """Cut down the imported dataset to only points with no errors"""
        if self.goodmask==None:
            print "Warning! No error column found during database import! All \
points will be assumed to be error-free."
        else:
            print "Removing records flagged as errornous from dataset..."
            if np.sum(self.goodmask)==0:
               raise ValueError("No good points left after application of mask! Nothing left to plot!")
            self.dset = self.dset[self.goodmask]
            if self.likedata!=None: self.likedata = self.likedata[self.goodmask]
            if self.probdata!=None: self.probdata = self.probdata[self.goodmask]
            self.goodmask = None #No more error rows in dataset!

    def removeallnonfinite(self):
        """Deletes all records from the dataset in which ANY field contains a
           non-finite value"""
        print "Checking dataset for any non-finite values..."
        fields = self.dset.dtype.names
        goodmask = np.ones((len(self.dset),), dtype=np.bool) 
        for field in fields:
          if field!='errors':
            badmask = np.logical_not(np.isfinite(self.dset[field]))
            if np.any(badmask):
              totbad = np.sum(badmask)
              print '{0} bad values found in field {1}...'.format(\
                                                                  totbad,field)
              goodmask = np.logical_and(goodmask, np.logical_not(badmask))
        print 'Removed total of {0} records with errornous entries'.format(
                    len(goodmask)-np.sum(goodmask))
        self.dset = self.dset[goodmask]
        if self.likedata!=None: self.likedata = self.likedata[goodmask]
        if self.probdata!=None: self.probdata = self.probdata[goodmask]

    def make2Dscatterplot(self,dataXY,axeslabels,filename,dataL=None):
        """Produce and export a scatter plot of the likelihood
        Args:
        dataXY - n*2 array of data to plot, each row is an (x,y) ordinate group
        dataL (optional) - n*1 array of likelihood data. By default is global likelihood.
        axeslabels - list of x,y axis labels
        filename - root name to give output file
        """
        if dataL==None: dataL = self.likedata
        # if STILL None throw an error
        if dataL==None: 
            raise ValueError('No default -2*loglikelihood column specified and none supplied!')
        
        #stitch together XY data and likelihood(chi2) column
        data = np.zeros((len(dataXY),3))
        data[:,:2] = dataXY
        data[:,2] = dataL

        #create plot        
        fig= plt.figure(figsize=self.plotsize)
        ax = fig.add_subplot(111)
        plot = chi2scatplot(ax,data,labels=axeslabels)
        fig.subplots_adjust(bottom=0.15)
        fig.subplots_adjust(left=0.15)
        fig.subplots_adjust(right=.99)
        cax,kw = colorbar.make_axes(ax, orientation='vertical', shrink=0.7, pad=0.04)
        cbar = fig.colorbar(plot, ax=ax, cax=cax, ticks=[0, 1, 2, 3, 4, 5], **kw)
        cbar.ax.set_yticklabels(['0','1','4','9','16',r'$\geq$25'])# vertically oriented colorbar
        cbar.ax.set_title("$\Delta\chi^2$")
        fig.savefig("{0}/{1}-scat.png".format(self.outdir,filename),dpi=(800/8))
        plt.close() #Must do this after EVERY plot!!! Or else memory will leak all over the place!
        
    def make2Dlogscatterplot(self,dataXY,axeslabels,filename,dataL=None,logaxis='y'):
        """Produce and export a scatter plot of the likelihood
        Args:
        dataXY - n*2 array of data to plot, each row is an (x,y) ordinate group
        dataL (optional) - n*1 array of likelihood data. By default is global likelihood.
        axeslabels - list of x,y axis labels
        filename - root name to give output file
        logaxis - string of axes to make 'log', e.g. 'x' for x axis, 'xy' for both etc.
        """
        if dataL==None: dataL = self.likedata
        # if STILL None throw an error
        if dataL==None: 
            raise ValueError('No default -2*loglikelihood column specified and none supplied!')
        
        #stitch together XY data and likelihood(chi2) column
        data = np.zeros((len(dataXY),3))
        data[:,:2] = dataXY
        data[:,2] = dataL

        #create plot        
        fig= plt.figure(figsize=self.plotsize)
        ax = fig.add_subplot(111)
        plot = chi2logscatplot(ax,data,labels=axeslabels,logaxis=logaxis)
        fig.subplots_adjust(bottom=0.15)
        fig.subplots_adjust(left=0.15)
        fig.subplots_adjust(right=.99)
        cax,kw = colorbar.make_axes(ax, orientation='vertical', shrink=0.7, pad=0.04)
        cbar = fig.colorbar(plot, ax=ax, cax=cax, ticks=[0, 1, 2, 3, 4, 5], **kw)
        cbar.ax.set_yticklabels(['0','1','4','9','16',r'$\geq$25'])# vertically oriented colorbar
        cbar.ax.set_title("$\Delta\chi^2$")
        fig.savefig("{0}/{1}-scat.png".format(self.outdir,filename),dpi=(800/8))
        plt.close() #Must do this after EVERY plot!!! Or else memory will leak all over the place!
        
    def make2Dprofileplot(self,dataXY,axeslabels,filename,dataL=None):
        """Produce and export a binned, profiled plot of the likelihood
        Args:
        dataXY - n*2 array of data to plot, each row is an (x,y) ordinate group
        dataL (optional) - n*1 array of likelihood data. By default is -2 * log (global likelihood).
        axeslabels - list of x,y axis labels
        filename - root name to give output file
        """
        if dataL==None: dataL = self.likedata
        # if STILL None throw an error
        if dataL==None: 
            raise ValueError('No default -2*loglikelihood column specified and none supplied!')
            
        #stitch together XY data and likelihood(chi2) column
        data = np.zeros((len(dataXY),3))
        data[:,:2] = dataXY
        data[:,2] = dataL

        #create plot        
        fig= plt.figure(figsize=self.plotsize)
        ax = fig.add_subplot(111)
        try:
            plot = profplot(ax,data,labels=axeslabels)
        except ValueError, err:
            print "{0}: Error encountered during plot creation, aborting...".format(err)
            return -1
        fig.subplots_adjust(bottom=0.15)
        fig.subplots_adjust(left=0.15)
        fig.subplots_adjust(right=.99)
        cax,kw = colorbar.make_axes(ax, orientation='vertical', shrink=0.7, pad=0.04)
        cbar = fig.colorbar(plot, ax=ax, cax=cax, ticks=[0, 1, 2, 3, 4, 5], **kw)
        cbar.ax.set_yticklabels(['0','1','4','9','16',r'$\geq$25'])# vertically oriented colorbar
        cbar.ax.set_title("$\Delta\chi^2$")
        fig.savefig("{0}/{1}-prof.pdf".format(self.outdir,filename),dpi=(800/8))
        plt.close() #Must do this after EVERY plot!!! Or else memory will leak all over the place!

    def make2Dposteriorplot(self,dataXY,axeslabels,filename,dataP=None):
        """Produce and export a binned, profiled plot of the likelihood
        Args:
        dataXY - n*2 array of data to plot, each row is an (x,y) ordinate group
        dataP (optional) - n*1 array of posterior data. By default is original posterior column.
        axeslabels - list of x,y axis labels
        filename - root name to give output file
        """
        if dataP==None: dataP = self.probdata
        # if STILL None throw an error
        if dataP==None: 
            raise ValueError('No default posterior column specified and none supplied!')
            
        #stitch together XY data and likelihood(chi2) column
        data = np.zeros((len(dataXY),3))
        data[:,:2] = dataXY
        data[:,2] = dataP

        #create plot        
        fig= plt.figure(figsize=self.plotsize)
        ax = fig.add_subplot(111)
        try:
            plot = margplot(ax,data,labels=axeslabels)
        except ValueError, err:
            print "{0}: Error encountered during plot creation, aborting...".format(err)
            return -1
        fig.subplots_adjust(bottom=0.15)
        fig.subplots_adjust(left=0.15)
        fig.subplots_adjust(right=.99)
        #print dir(plot)
        zdata = plot.get_array()
        minP = min(zdata.flatten())
        maxP = max(zdata.flatten())
        step = (maxP -minP) / 5
        ticks = np.arange(minP,maxP+2*step,step)
        print ticks
        print np.arange(0.,1.+2./5,1./5)
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_yticklabels(np.arange(0.,1.+2./5,1./5))
        cbar.ax.set_ylabel("Rel. post. density")
        fig.savefig("{0}/{1}-post.pdf".format(self.outdir,filename),dpi=(800/8))
        plt.close() #Must do this after EVERY plot!!! Or else memory will leak all over the place!

    def make1Dposteriorplot(self,dataX,axeslabel,filename,dataP=None):
        """Produce and export a binned, marginalised plot of the posterior density
        Args:
        dataX - 1D array of data to plot on x axis
        dataP (optional) - 1D array of posterior data. By default is original posterior column.
        axeslabel - x axis label
        filename - root name to give output file
        """
        if dataP==None: dataP = self.probdata
        # if STILL None throw an error
        if dataP==None: 
            raise ValueError('No default posterior column specified and none supplied!')
            
        #stitch together X data and posterior column
        data = np.zeros((len(dataX),2))
        print dataX[:10]
        data[:,0] = dataX
        data[:,1] = dataP

        #create plot        
        fig= plt.figure(figsize=self.plotsize)
        ax = fig.add_subplot(111)
        try:
            plot = margplot1D(ax,data,labels=axeslabel)
        except ValueError, err:
            print "{0}: Error encountered during plot creation, aborting...".format(err)
            return -1
        fig.subplots_adjust(bottom=0.15)
        fig.subplots_adjust(left=0.15)
        fig.subplots_adjust(right=.99)
        #print dir(plot)
        
        fig.savefig("{0}/{1}-1Dpost.pdf".format(self.outdir,filename),dpi=(800/8))
        plt.close() #Must do this after EVERY plot!!! Or else memory will leak all over the place!
        

    def maketimingplot(self):
        """Produce and export an assortment of histograms containing
        timing data (ONLY WORKS IF LINKED DATASET IS A TIMING DATASET
        """
        
        fig = plt.figure(figsize=(8,12))
        ax = fig.add_subplot(611)
        ax1 = fig.add_subplot(612)
        ax2 = fig.add_subplot(613)
        ax3 = fig.add_subplot(614)
        ax4 = fig.add_subplot(615)
        ax5 = fig.add_subplot(616)
        
        times = ['looptime','samplertime', 'liketime']
        colorsavail = ['blue','red','green','orange','purple','brown']
        colorslist = zip(colorsavail[:len(times)],colorsavail[:len(times)])
        
        print 'Creating program runtime fraction histograms...'
        for name,colors in zip(times,colorslist):
            timehist(self.dset[name],ax,ax1,labels=[name,name+' fraction'],colors=colors)
        
        print 'Creating program runtime fraction histograms'    
        progs = [prog for prog in self.dset.dtype.names if 'prog' in prog]
        colorslist = zip(colorsavail[:len(progs)],colorsavail[:len(progs)])
        for prog,colors in zip(progs,colorslist):
            timehist(self.dset[prog],ax2,ax3,labels=[prog,prog+' fraction'],colors=colors)
            
        datalen = self.dset.shape[0]
        looptime = self.dset['looptime']
        ax4.scatter(range(datalen),looptime,label='looptime',s=1,lw=0)
        ax4.set_xlim(0,datalen)
        ax4.set_ylim(bottom=0)
        bins=1000
        if datalen<10*bins: bins = datalen/10.    # make sure we don't have more bins than data
        w = np.round(datalen/bins)   #number of loops to average over
        movingaverage = []
        for i in range(0,datalen,w):
            movingaverage += [np.average(looptime[i:i+w])]
        ax5.plot(range(len(movingaverage)),movingaverage,label='looptime (run. avg.)')
        ax5.set_xlim(0,len(movingaverage))
        ax5.set_ylim(bottom=0)
        
        axlist=[ax,ax1,ax2,ax3,ax4,ax5]
        for axis in axlist:
            leg = axis.legend()
            leg.get_frame().set_alpha(0.5)
        
        fig.suptitle('Timing histograms')
        fig.savefig("{0}/timinginfo.png".format(self.outdir),dpi=(800/8))
        plt.close() #Must do this after EVERY plot!!! Or else memory will leak all over the place!

#---------------------Even higher level functions!----------------------
    def makechecklikefuncplots(self):
        """Make a series of plots useful for visually confirming that
        the various likelihood functions are working as expected"""
        
        allfields = self.dsets.dtype.names

        #Automation for simple likelihood function checking plots
        parlistlikes = [[find(lambda x: par.split('-')[1] in x and 'logl-' not in x, allfields), par]
                            for par in allfields if 'logl-' in par]             #bit hard to read, but gets observables fieldnames and the fieldnames of their matching likelihood value

        n = np.round(len(self.dset[self.likepar])*0.1)      #cut off this many of the lowest likelihood points to improve the visibility of some of the likelihoods with huge outliers
        for obs,like in parlistlikes:
            dataXY = self.getcols([obs,like])
            dataXY[:,1] = np.exp(dataXY[:,1])   #likelihood
            dataL = -2*self.dset[like]    #Delta chi^2
            self.make2Dscatterplot(dataXY[n:],[obs, like],obs+'likecheck',dataL[n:])
    
    def make2Dplotgrid(self,dataALL,axeslabels,filename,dataL=None,
                        dataP=None,plottype='scatter',colors=None,sortcolors=True,overlay=False,figaxes=None,alphas=1):
        """Produce and export a grid of scatter plots of the likelihood. This is 
        a nice way to see many projections of the parameter space on the same
        plot.
        Args:
        dataALL - n*m array of data to plot. Each of n rows is a vector of 
            length m, containing the various parameter columns to be plotted.
        axeslabels - list of axis labels corresponding to order of parameters in
            columns of 'data'.
        filename - root name to give output file
        dataL (optional) - n*1 array of likelihood data. By default is global 
            likelihood.
        dataP (optional) - n*1 array of posterior mass data. By default is 
            original scan posterior.
        plottype (optional) - which sort of plot to put in the grid. Current 
            valid options are: 'scatter', 'profile', 'posterior'. Default is
            'scatter'.
            NEW: new plottype 'customcolors' available. Must supply a vector
            of colours to use of length n in 'colors'
        """
        if plottype=='scatter' or plottype=='profile':
            if dataL==None: dataL = self.likedata
            # if STILL None throw an error
            if dataL==None: 
                raise ValueError('No default -2*loglikelihood column specified \
and none supplied!')
        elif plottype=='posterior':   
            if dataP==None: dataP = self.probdata
            # if STILL None throw an error
            if dataP==None: 
                raise ValueError('No default posterior column specified and \
none supplied!')
        elif plottype=='customcolors':
            if colors==None:
                raise ValueError('To use plottype "customcolors" please supply a \
2D array of RBG values (i.e. n*3 array) in the argument "colors"')
        else:
            raise ValueError("Invalid value provide for plottype \
option. Currently valid values are 'scatter', 'profile' or 'posterior' (got \
'{0}')".format(plottype))
                
        n = len(axeslabels)
        if n!=len(dataALL[0,:]):
            raise ValueError('Length of axes labels list does not match number \
of data columns supplied!')
   
        #create plot        
        #fig= plt.figure(figsize=self.gridplotsize)
                
        # Need to cycle through the various combinations of pairs of parameters,
        # and produce a scatter plot for each of them.
        if figaxes==None: 
           fig,axes = plt.subplots(n-1,n-1,figsize=self.gridplotsize)
           for ax in axes.flatten():
              ax.axis('off')
        else:
           fig,axes = figaxes

        for j in range(1,n):
            for i in range(0,j):
                #print i,j,i+(n-1)*(j-1)
                print axeslabels[i],axeslabels[j]
                #stitch together XY data and likelihood(chi2) (or post.) column
                data = np.zeros((len(dataALL),3))
                data[:,:2] = dataALL[:,[i,j]]

                #create plot        
                #ax = fig.add_subplot(n-1,n-1,(i+1)+(n-1)*(j-1))
                ax = axes[j-1,i]
                ax.axis('on')
                ax.margins(0,0)

                if plottype=='scatter':
                    data[:,2] = dataL
                    plot = chi2scatplot(ax,data,labels=[axeslabels[i],axeslabels[j]],overlay=overlay)
                elif plottype=='profile':
                    data[:,2] = dataL
                    plot = profplot(ax,data,labels=[axeslabels[i],axeslabels[j]],
                                        nxbins=75,nybins=75)
                elif plottype=='posterior':
                    data[:,2] = dataP                                   
                    plot = margplot(ax,data,labels=[axeslabels[i],axeslabels[j]],
                                        nxbins=75,nybins=75)
                elif plottype=='customcolors':
                    if sortcolors:
                       #draw the higher likelihood points on top
                       sorti=np.argsort(dataL)[::-1]
                       # No alpha argument allowed: alpha channel should be part of 'colors' array if you want it.
                       plot = ax.scatter(data[sorti,0],data[sorti,1],s=1,lw=0,c=colors[sorti])     
                    else:
                       # Shuffle the data and color arrays in sync, so that we
                       # don't bias the order the points are drawn by the plotter.
                       rng_state = np.random.get_state()
                       np.random.shuffle(data)
                       np.random.set_state(rng_state)
                       np.random.shuffle(colors)
                       plot = ax.scatter(data[:,0],data[:,1],s=1,lw=0,c=colors,alpha=alphas)
                    if not overlay: ax.set_xlabel(axeslabels[i])
                    if not overlay: ax.set_ylabel(axeslabels[j])
                else:
                    raise ValueError("Invalid value provide for plottype \
option. Currently valid values are 'scatter', 'profile' or 'posterior' (got \
'{0}')".format(plottype))
                
                if not overlay:
                   #cut off tick labels that overlap between subplots
                   nxbins = len(ax.get_xticklabels())
                   nybins = len(ax.get_yticklabels())
                   ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=nxbins, prune='upper'))
                   ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=nybins, prune='upper'))
                   
                   #rotate x-tick labels so we can read them...
                   labels = ax.get_xticklabels() 
                   for label in labels: 
                       label.set_rotation(-90)
    
                   #remove the tick and axis labels from 'interior' plots
                   if j<(n-1): 
                       ax.set_xticklabels([])
                       ax.set_xlabel('')
                   if i>0: 
                       ax.set_yticklabels([])
                       ax.set_ylabel('')
                   
                """
                fig.subplots_adjust(bottom=0.15)
                fig.subplots_adjust(left=0.15)
                fig.subplots_adjust(right=.99)
                
                cax,kw = colorbar.make_axes(ax, orientation='vertical', shrink=0.7, pad=0.04)
                cbar = fig.colorbar(plot, ax=ax, cax=cax, ticks=[0, 1, 2, 3, 4, 5], **kw)
                cbar.ax.set_yticklabels(['0','1','4','9','16',r'$\geq$25'])# vertically oriented colorbar
                cbar.ax.set_title("$\Delta\chi^2$")
                """   
        
        plt.tight_layout()
        
        # Cut out space between subplots
        fig.subplots_adjust(wspace=0,hspace=0)
   
        # Save the plots
        #fig.savefig("{0}/{1}-grid{2}.png".format(self.outdir,filename,plottype),dpi=(800/8))
        #plt.close() #Must do this after EVERY plot!!! Or else memory will leak all over the place!

        return fig,axes
        
 
# Generalisation of make2Dplotgrid, to accept seperate lists for X and Y parameters to plot.
    def make2DplotgridXY(self,(dataX,dataY),(axeslabelsX,axeslabelsY),filename,dataL=None,
                        dataP=None,plottype='scatter',colors=None,sortcolors=True):
        """Produce and export a grid of scatter plots of the likelihood. This is 
        a nice way to see many projections of the parameter space on the same
        plot.
        Args:

        dataX - n*nX array of data to plot. Each of n rows is a vector of 
            length nX, containing the various parameter columns to be plotted.
        dataY - n*nY array of data to plot. Each of n rows is a vector of 
            length nY, containing the various parameter columns to be plotted.

        axeslabelsX - list of axis labels corresponding to order of parameters in
            columns of 'dataX'.
        axeslabelsY - list of axis labels corresponding to order of parameters in
            columns of 'dataY'. 

        filename - root name to give output file
        dataL (optional) - n*1 array of likelihood data. By default is global 
            likelihood.
        dataP (optional) - n*1 array of posterior mass data. By default is 
            original scan posterior.
        plottype (optional) - which sort of plot to put in the grid. Current 
            valid options are: 'scatter', 'profile', 'posterior'. Default is
            'scatter'.
            NEW: new plottype 'customcolors' available. Must supply a vector
            of colours to use of length n in 'colors'
        """
        if plottype=='scatter' or plottype=='profile':
            if dataL==None: dataL = self.likedata
            # if STILL None throw an error
            if dataL==None: 
                raise ValueError('No default -2*loglikelihood column specified \
and none supplied!')
        elif plottype=='posterior':   
            if dataP==None: dataP = self.probdata
            # if STILL None throw an error
            if dataP==None: 
                raise ValueError('No default posterior column specified and \
none supplied!')
        elif plottype=='customcolors':
            if colors==None:
                raise ValueError('To use plottype "customcolors" please supply a \
2D array of RBG values (i.e. n*3 array) in the argument "colors"')
        else:
            raise ValueError("Invalid value provide for plottype \
option. Currently valid values are 'scatter', 'profile' or 'posterior' (got \
'{0}')".format(plottype))
                
        nX = len(axeslabelsX)
        if nX!=len(dataX[0,:]):
            raise ValueError('Length of X axes labels list does not match number \
of X data columns supplied!')
  
        nY = len(axeslabelsY)
        if nY!=len(dataY[0,:]):
            raise ValueError('Length of Y axes labels list does not match number \
of Y data columns supplied!')
   
        #create plot        
        fig= plt.figure(figsize=self.gridplotsize)
                
        # Need to cycle through the various combinations of pairs of parameters,
        # and produce a scatter plot for each of them.
        for j in range(0,nY):
            for i in range(0,nX):
                #print i,j,i+(n-1)*(j-1)
                print axeslabelsX[i],axeslabelsY[j]
                #stitch together XY data and likelihood(chi2) (or post.) column
                data = np.zeros((len(dataX),3)) #should check len(dataX)=len(dataY)...
                data[:,0] = dataX[:,i]
                data[:,1] = dataY[:,j]  

                #create plot       
                print 'plot number', (i+1)+nX*j 
                ax = fig.add_subplot(nY,nX,(i+1)+nX*j)
                if plottype=='scatter':
                    data[:,2] = dataL
                    plot = chi2scatplot(ax,data,labels=[axeslabelsX[i],axeslabelsY[j]])
                elif plottype=='profile':
                    data[:,2] = dataL
                    plot = profplot(ax,data,labels=[axeslabelsX[i],axeslabelsY[j]],
                                        nxbins=75,nybins=75)
                elif plottype=='posterior':
                    data[:,2] = dataP                                   
                    plot = margplot(ax,data,labels=[axeslabelsX[i],axeslabelsY[j]],
                                        nxbins=75,nybins=75)
                elif plottype=='customcolors':
                    if sortcolors:
                       #draw the higher likelihood points on top
                       sorti=np.argsort(dataL)[::-1]
                       plot = ax.scatter(data[sorti,0],data[sorti,1],s=1,lw=0,c=colors[sorti],alpha=0.6)     
                    else:
                       # Shuffle the data and color arrays in sync, so that we
                       # don't bias the order the points are drawn by the plotter.
                       rng_state = np.random.get_state()
                       np.random.shuffle(data)
                       np.random.set_state(rng_state)
                       np.random.shuffle(colors)
                       plot = ax.scatter(data[:,0],data[:,1],s=1,lw=0,c=colors,alpha=0.6)
                else:
                    raise ValueError("Invalid value provide for plottype \
option. Currently valid values are 'scatter', 'profile' or 'posterior' (got \
'{0}')".format(plottype))
                
                #cut off tick labels that overlap between subplots
                nxbins = len(ax.get_xticklabels())
                nybins = len(ax.get_yticklabels())
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=nxbins, prune='upper'))
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=nybins, prune='upper'))
                
                #rotate x-tick labels so we can read them...
                labels = ax.get_xticklabels() 
                for label in labels: 
                    label.set_rotation(-90)
    
                #remove the tick and axis labels from 'interior' plots
                print i,nX, j, nY
                if j<(nY-1): 
                    ax.set_xticklabels([])
                    ax.set_xlabel('')
                if i>0: 
                    ax.set_yticklabels([])
                    ax.set_ylabel('')
                
                """
                fig.subplots_adjust(bottom=0.15)
                fig.subplots_adjust(left=0.15)
                fig.subplots_adjust(right=.99)
                
                cax,kw = colorbar.make_axes(ax, orientation='vertical', shrink=0.7, pad=0.04)
                cbar = fig.colorbar(plot, ax=ax, cax=cax, ticks=[0, 1, 2, 3, 4, 5], **kw)
                cbar.ax.set_yticklabels(['0','1','4','9','16',r'$\geq$25'])# vertically oriented colorbar
                cbar.ax.set_title("$\Delta\chi^2$")
                """   
        
        plt.tight_layout()
        
        # Cut out space between subplots
        fig.subplots_adjust(wspace=0,hspace=0)
   
        # Save the plot
        fig.savefig("{0}/{1}-grid{2}.png".format(self.outdir,filename,plottype),dpi=(800/8))
        plt.close() #Must do this after EVERY plot!!! Or else memory will leak all over the place!

#Modified version of dataset analysis class for dealing with ev.dat and live point files
#(combines into one dataset of same form as .txt dataset)

#DON'T USE THIS ANYMORE: OBSOLETE. CREATE 'evlive' DATASET INSTEAD
class LinkEvDataSetForAnalysis(LinkDataSetForAnalysis):
    
    #------Parameters set by __init__-----------------
    nlive = None
    Zdata = None
          
    def __init__(self,evdsetIN,livedsetIN,outdir,allpars,loglpar,logwpar):
        """Initialise link to datasets, create output directory and check
        that requested data columns are present in dataset.
            Args:
        evdsetIN - hdf5 dataset (or structured array) containing ev.dat data to analyse
        livedsetIN - hdf5 dataset (or structured array) containing live point data to analyse
        outdir - Path to directory to store results
        allpars - list of fieldnames of columns to to be extracted from datasets
        loglpar - loglikelihood fieldname
        logwpar - prior mass fieldname
        """
        likepar = 'neg2LogL'    #name to give chi2 column we are going to create
        probpar = 'P'           #   "   "   posterior   "   "
        self.likepar = likepar
        self.probpar = probpar
        self.outdir = outdir
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print "Storing plots in {0}...".format(outdir)

        evfieldnames = evdsetIN.dtype.names
        livefieldnames = livedsetIN.dtype.names
        
        allpars = list(set(allpars + [loglpar, logwpar]))   #remove duplicates
        
        #-------Verify requested fieldnames-----------
        #ev.dat
        evparsVERIFIED = [name for name in allpars if name in evfieldnames] #make sure parameters specified by 'allpars' exist in dataset
        nreq = len(allpars)
        nfound = len(evparsVERIFIED)
        if nfound != nreq:
            print "WARNING! Not all fieldnames requested could be found in ev.dat dataset! Continuing with those found. \
            (# requested = {0}, # found = {1}, # missing = {2})".format(nreq, nfound, nreq - nfound)
            print "Fieldnames not found: {0}".format([name for name in allpars if name not in evparsVERIFIED])
        
        #live
        livepars = [v for v in allpars if v!=logwpar]   #don't try and extract logwpar; doesn't exist.
        liveparsVERIFIED = [name for name in livepars if name in livefieldnames] #make sure parameters specified by 'allpars' exist in dataset
        nreq = len(livepars)
        nfound = len(liveparsVERIFIED)
        if nfound != nreq:
            print "WARNING! Not all fieldnames requested could be found in live dataset! Continuing with those found. \
            (# requested = {0}, # found = {1}, # missing = {2})".format(nreq, nfound, nreq - nfound)
            print "Fieldnames not found: {0}".format([name for name in livepars if name not in liveparsVERIFIED])
            
        #import subset of data
        evdset = evdsetIN[:][evparsVERIFIED]
        livedset = livedsetIN[:][liveparsVERIFIED]
        #--------------------------------------------
        
        #-------Compute posterior from likelihood and prior mass--------
        evlogw = evdset[logwpar]
        evlogL = evdset[loglpar]
        evPdata = np.exp(evlogL + evlogw) #exp(logL + logw) = likelihood * prior (unnormalised posterior)
        Pleft = 1 - sum(np.exp(evlogw))
        print 'Pleft: ',Pleft, sum(np.exp(evlogw))
        if Pleft < 0: Pleft = 0     #Negative prior volume left makes no sense, so assume that correct amount left is negligible.
        livelogL = livedset[loglpar]
        nlive = len(livelogL)
        self.nlive = nlive
        livePdata = np.exp(livelogL) * Pleft/len(livelogL)  # (unnormalised posterior for live points)

        post = np.append(evPdata,livePdata) #stitch posterior pieces together
        chi2 = -2*np.append(evlogL,livelogL) #stitch likelihood pieces together and convert to chi2
        logw = np.append(evlogw,np.zeros(nlive)+Pleft/nlive) #stitch prior mass pieces together
        #---------------------------------------------------------------
        
        #------Stitch datasets together-----------------------
        dset = np.zeros((len(evlogL)+nlive,), dtype=[(par,'<f4') for par in allpars+[likepar,probpar]]) #initialise
        for name in dset.dtype.names:
            if name not in loglpar+logwpar+likepar+probpar:
                dset[name] = np.append(evdset[name],livedset[name])
        dset[logwpar] = logw
        dset[probpar] = post
        
        #check if CCR prior used and correct likelihood.
        #CCRchi2 = -2 * np.array(evdset['BQloglikelihood'])
        #liveCCRchi2 = -2 * np.array(livedset['BQloglikelihood'])
        #CCRchi2 = np.append(CCRchi2,liveCCRchi2,axis=0)
        CCRchi2 = -2 * dset['BQloglikelihood']
        CCRprior = CCRchi2.any() #Check if any values for this 
                #column are non-zero. If they are, we used the CCR prior for this dataset.
        self.CCRprior = CCRprior
 
        #correct likelihood columns (subtract off (potential) CCR prior contributions)
        dset[likepar] = chi2 - CCRchi2
        dset[loglpar] = np.append(evdset[loglpar],livedset[loglpar]) + 0.5*CCRchi2
        self.dset = dset
        self.likedata = dset[likepar]
        
        #correct posterior column
        Z, dZ, logZ, dlogZ = computeevidence(post,dset[loglpar],nlive) #Compute evidence (need to normalise posterior data)
        #Z, dZ, logZ, dlogZ = self.getevidence()    #Compute evidence (need to normalise posterior data)
        dset[probpar] = post/Z
        self.probdata = dset[probpar]
        self.Zdata = Z, dZ, logZ, dlogZ
        
        print un2str(logZ, dlogZ)
        print un2str(*evdsetIN.attrs['logZ'])
        
        #----------------------------------------------------------
        
#=======================================================================
#       EVIDENCE ANALYSIS TOOLS
#=======================================================================
# This set of tools is designed for doing the probability theory
# calculations associated with model selection and producing associated 
# graphs.

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

    return B, LR, OH0, OH1  #B is the log Bayes factor, LR the log likelihood ratio, OH0/1 the log occam factors
    
"""
datalist=['preLEP','LEP','SUSY','higgs']
datachanges=[['preLEP->LEP','preLEP->SUSY','preLEP->higgs'],\
            ['preLEP->LEP','LEP->SUSY','SUSY->higgs']]
    
altscenarios=['SM']   #alternate model (SM)
scenarios=['log','CCR','no(g-2)log','no(g-2)CCR'] #CMSSM
"""
def evsummary(dsetevs,scenarios,altscenarios,datalist):
    """Produces a summary of the evidence ratios associated with the
    supplied evidence data
    
    Args:
        dsetevs - nested dictionary of evidence values and maximum log
            likelihood value, with the following structure:
                { 'model 1' : {'data state 1' : ([logZ,dlogZ], maxlogl),
                               'data state 2' : ([logZ,dlogZ], maxlogl),
                               ...
                               },
                  'model 2' : {'data state 1' : ([logZ,dlogZ], maxlogl),
                               'data state 2' : ([logZ,dlogZ], maxlogl),
                               ...
                               }
                 ...
                }
        scenarios - list of keys in first layer of dsetevs, representing
            a set of a models to be tested, each in turn, against the
            models listed in 'altscenarios'. e.g.
                ['model 1', 'model 2', ...]
        altscenarios - as above, but the hypotheseses competing against
            the first list:
                ['model 5', 'model 6', ...]
        datalist - a list of keys for entries of the second later of
            dsetevs, representing a sequence of experiments, or data
            sets. E.g. say for a straight line fit, element 1 is the 
            data set with say 3 data points, with element 2, 3, 4 etc
            being the same data plus new data points, e.g.
                ['data state 1', 'data state 2', 'data state 3' ...]
            The order is important as the results will be presented as
            a series of Bayesian updates in this sequence, from least
            information to most information.
    """
    print '----------------------------------------------------------------'
    print 'Belief change factors'
    print '----------------------------------------------------------------'
    
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
                print '{0:<10}-->{1:<10} ; logB:{2}, bits:{3}, B:{4}, OH0:{5}, OH1:{6}, LR:{7}'.format(\
                    dkinit,dk,\
                    un2str(B.nominal_value,B.std_dev(),2),\
                    un2str((np.log2(np.e)*B).nominal_value,(np.log2(np.e)*B).std_dev(),2),\
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
                print '{0:<10}-->{1:<10} ; logB:{2}, bits:{3}, B:{4}, OH0:{5}, OH1:{6}, LR:{7}'.format(\
                    dkprev,dk,\
                    un2str(B.nominal_value,B.std_dev(),2),\
                    un2str((np.log2(np.e)*B).nominal_value,(np.log2(np.e)*B).std_dev(),2),\
                    un2str(uexp(B).nominal_value,uexp(B).std_dev(),2),\
                    un2str(uexp(-OH0).nominal_value,uexp(-OH0).std_dev(),2),\
                    un2str(uexp(-OH1).nominal_value,uexp(-OH1).std_dev(),2),\
                    np.exp(LR)
                    )
        print '---------------------------------'
        return Bresults
        
#----------------Bayes factor breakdown bar chart-----------------------

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
    w = 1./5
    colors = ['r','b','g','m']

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
    
#COPIED DIRECTLY FROM makebayesfactorchart, and chopped out the occam factor part
def makebayesfactoronlychart(dsetevs,scenarios,altscenarios,datalist,Bresults):
    """a pair of charts illustrating the breakdown of the Bayes factor
    into likelihood ratio and Occam factor components"""
    
    #Need to add in likelihood ratio contributions to evidence. Only the higgs
    #search maximum likelihood values are different in the CMSSM and SM so we will just
    #add this one case in manually.
    
    #need to go through each 'scenario' and compute the likelihood ratios for each piece of data
    
    # B, LR, OH0, OH1
    # 0, 1, 2, 3
    
    HSM = altscenarios[0]  #SM
    Hlog = 'log'  #SM Occam factors are the same for all CMSSM scenarios so just pick one
    
    #PRETTY HACKY THING I AM DOING HERE, SO THAT I CAN IMMEDIATELY USE CODE FROM THE ORIGINAL summary.py
    datachanges=[[],['{0}->{1}'.format(datalist[i],datalist[i+1]) for i in range(len(datalist)-1)]]
    
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
    
    plotsize=(11,6)
    
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
    #fig.savefig("{0}/{1}.pdf".format(outdir,filename),dpi=(800/8))
    fig.savefig("{0}.pdf".format(filename),dpi=(800/8))

def makebayesfactorchart(dsetevs,scenarios,altscenarios,datalist,Bresults):
    """a pair of charts illustrating the breakdown of the Bayes factor
    into likelihood ratio and Occam factor components"""
    
    #Need to add in likelihood ratio contributions to evidence. Only the higgs
    #search maximum likelihood values are different in the CMSSM and SM so we will just
    #add this one case in manually.
    
    #need to go through each 'scenario' and compute the likelihood ratios for each piece of data
    
    # B, LR, OH0, OH1
    # 0, 1, 2, 3
    
    HSM = altscenarios[0]  #SM
    Hlog = 'log'  #SM Occam factors are the same for all CMSSM scenarios so just pick one
    
    #PRETTY HACKY THING I AM DOING HERE, SO THAT I CAN IMMEDIATELY USE CODE FROM THE ORIGINAL summary.py
    datachanges=[[],['{0}->{1}'.format(datalist[i],datalist[i+1]) for i in range(len(datalist)-1)]]
    
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
    
    plotsize=(11,6)
    
    fig= plt.figure(figsize=plotsize)

    #---------DELETED "OCCAM+LR" BREAKDOWN PLOT-------------    
    
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
    
    ax = fig.add_subplot(111)#,axisbg='0.9')
    
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
    
    filename = 'BayesFactors'
    #fig.savefig("{0}/{1}.pdf".format(outdir,filename),dpi=(800/8))
    fig.savefig("{0}.pdf".format(filename),dpi=(800/8))


