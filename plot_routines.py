# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:49:42 2015

@author: traveller
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.distributions as dist
from pdb import set_trace
import copy

def weighted_percentile(quant, percs, weights):
    
    sort_inds = np.argsort(quant)    
    cumweightsum = np.cumsum(np.array(weights)[sort_inds])
    cumweightsum = cumweightsum / cumweightsum[-1]
    cws_withedge = np.concatenate(([0], cumweightsum))
    cws_central = cws_withedge[:-1] + (cws_withedge[1:]-cws_withedge[:-1])/2.0

    wpercs = np.interp(percs/100.0, cws_central, np.array(quant)[sort_inds])        
    #set_trace()

    
    return wpercs

def plot_distribution(x,y,nbins=10, xrange=None, percent=50.0, uncertainty=False, 
                      scatter=True,plot_at_median=False, show_points_below = -1, 
                      dashed_below = 10, label='', medlinestyle = '-', logbins = False,
                      dashedline=True, alphavec = (0.3,0.15,0.7,0.25),
                      refdist=None, plot=True, simple_scatter = False, weights = None, pointalpha = 1.0, **kwargs):

    if refdist == None:
        if simple_scatter:
            refdist_percs = np.zeros(nbins)
            n_eff_ref = None
        else:
            refdist_percs = np.zeros((nbins,5))
            n_eff_ref = None
    else:
        refdist_percs = refdist[0]
        n_eff_ref = refdist[1]


    lowlim = 50.0-percent/2.0
    highlim = 50.0+percent/2.0
    percvec = np.array((percent, lowlim, highlim, 15.9, 84.1))    
    
    if xrange is None:
        xmin = np.min(x)
        xmax = np.max(x)
    else:
        xmin = xrange[0]
        xmax = xrange[1]

    if logbins:
        binedges = np.logspace(np.log10(xmin), np.log10(xmax), num = nbins+1, endpoint = True)
    else:
        binedges = np.linspace(xmin, xmax, num=nbins+1, endpoint=True)

    bincen = binedges[:-1]+(binedges[1:]-binedges[:-1])/2.0

    nums = np.zeros(nbins)
    percs = np.zeros((nbins, 5))                       
    n_eff = np.zeros(nbins)

    if 'color' in kwargs:
        pltcol = kwargs['color']
    else:
        pltcol = 'blue'
        
    if 'linewidth' in kwargs:
        pltlw = kwargs['linewidth']
        print(f'Linewidth={pltlw}')
        del kwargs['linewidth']
    else:
        pltlw = 2.0
        
    if 's' in kwargs:
        plts = kwargs['s']
        del kwargs['s']
    else:
        plts = 10
                        
    for ibin in range(nbins):
 
        ind_curr = (np.nonzero((x >= binedges[ibin]) & (x < binedges[ibin+1])))[0]
        nums[ibin] = len(ind_curr)
            
        if len(ind_curr) > 0:
            if weights is None:
                percs[ibin,:] = np.percentile(y[ind_curr], percvec)            
                bincen[ibin]=np.median(x[ind_curr])
                n_eff[ibin] = len(ind_curr)
            else:
                percs[ibin,:] = weighted_percentile(y[ind_curr], percvec, weights = weights[ind_curr])
                bincen[ibin] = weighted_percentile(x[ind_curr], 50.0, weights = weights[ind_curr])
                weights_rel = weights[ind_curr]/np.max(weights[ind_curr])
                n_eff[ibin] = np.sum(weights_rel)


            
        if not isinstance(pltcol, str):
#            pltcol = np.array([pltcol])
            pltcol=np.array([pltcol])            

        if plot == True:
            if show_points_below != -1:    
                if len(ind_curr) < show_points_below:
                    plt.scatter(x[ind_curr], y[ind_curr]-refdist_percs[ibin,0],
                                facecolors='none', alpha=pointalpha,
                                s=plts, edgecolors=pltcol, linestyle=medlinestyle,
                                linewidth=pltlw)
                    #set_trace()
                    print("Plotting", len(ind_curr), "points in bin", ibin,
                          "[", binedges[ibin], "--", binedges[ibin+1], "] in",
                          kwargs['color'])
                else:
                    print("Bin", ibin, "[", binedges[ibin], "--", binedges[ibin+1], "] in",
                          kwargs['color'], "has", len(ind_curr), "galaxies (N_eff = {:.2f})" .format(n_eff[ibin]))

    # Subtract REFDIST from all rows in percs:     
    refdist_median = refdist_percs[:,0] 
    percs -= np.transpose(np.array([refdist_median,]*5))
 
    if plot == True:
        ind_plot = (np.nonzero(nums > 0))[0]    
    
        """
        if dashedline == True:
            plt.plot(bincen[ind_plot], percs[ind_plot, 0], linestyle = '--', linewidth = 1.5, **kwargs)        
        """
            
        if refdist is None:
            ind_plot = (np.nonzero(n_eff >= dashed_below))[0]    
        else:
            ind_plot = np.nonzero((n_eff >= dashed_below) & (n_eff_ref >= dashed_below))[0]
     
        plt.plot(bincen[ind_plot], percs[ind_plot, 0], linestyle = medlinestyle,label = label,
                 linewidth=pltlw, **kwargs)       

        lowunc = percs[ind_plot, 0] - (percs[ind_plot, 0] - percs[ind_plot,3])/np.sqrt(n_eff[ind_plot])
        highunc = percs[ind_plot, 0] + (-percs[ind_plot, 0] + percs[ind_plot,4])/np.sqrt(n_eff[ind_plot])

        if not simple_scatter and n_eff_ref is not None:
            delta_lowunc_ref = (refdist_percs[ind_plot, 0] - refdist_percs[ind_plot,3])/np.sqrt(n_eff_ref[ind_plot])
            delta_highunc_ref = (-refdist_percs[ind_plot, 0] + refdist_percs[ind_plot,4])/np.sqrt(n_eff_ref[ind_plot])

            lowunc = percs[ind_plot,0] - np.sqrt((percs[ind_plot,0]-lowunc)**2 + delta_lowunc_ref**2)
            highunc = percs[ind_plot,0] + np.sqrt((highunc-percs[ind_plot,0])**2 + delta_highunc_ref**2)

        if uncertainty == True:
            plt.fill_between(bincen[ind_plot], lowunc, highunc, alpha = alphavec[0], **kwargs)
            plt.plot(bincen[ind_plot], lowunc, linewidth=0.5, alpha=alphavec[2], **kwargs)
            plt.plot(bincen[ind_plot], highunc, linewidth=0.5, alpha=alphavec[2], **kwargs)
        
        if scatter == True:
            plt.fill_between(bincen[ind_plot], percs[ind_plot, 1], percs[ind_plot,2], alpha = alphavec[1], **kwargs)
            plt.plot(bincen[ind_plot], percs[ind_plot, 1], linewidth=0.5, alpha=alphavec[3], **kwargs)        
            plt.plot(bincen[ind_plot], percs[ind_plot, 2], linewidth=0.5, alpha=alphavec[3], **kwargs) 
            
            if len(ind_plot) == 1:
                plt.fill_between((bincen[ind_plot[0]]-0.05,bincen[ind_plot[0]]+0.05),(percs[ind_plot[0],1],percs[ind_plot[0],1]),(percs[ind_plot[0],2],percs[ind_plot[0],2]),alpha=alphavec[1], **kwargs)

    return percs, n_eff, bincen
            
            
            
            
def plot_nums(x, binedges, color='black', **kwargs):
    num_gal = np.zeros(5)
    med_mstar = np.zeros(5)-1

    delta_mstar = binedges[1]-binedges[0]

    for imstar in range(len(binedges)-1):
        ind_mstar = (np.nonzero((x >= binedges[imstar]) & (x < binedges[imstar+1])))[0]
        num_gal[imstar] = len(ind_mstar)
        
        if num_gal[imstar] > 0:
            med_mstar[imstar] = np.median(x[ind_mstar])
        else:
            med_mstar[imstar] = binedges[imstar] + delta_mstar/2

    plt.plot(med_mstar, num_gal, color=color, marker = 'o',**kwargs)
    
    
    
def plot_fraction_above_threshold(x,y,thresh,nbins=10, xrange=None, 
          uncertainty=False, dashed_below = 10, label='', medlinestyle = '-', 
          dashedline=False, alphaunc = 0.15 ,ci = 0.68,
          plot_med_line=True, mode = 'above', medlinewidth = 1.5, plot = True, 
          retvals = False, **kwargs):    
   
    if xrange is None:
        xmin = np.min(x)
        xmax = np.max(x)
    else:
        xmin = xrange[0]
        xmax = xrange[1]
                 
    binedges = np.linspace(xmin, xmax, num=nbins+1, endpoint=True)
    bincen = binedges[0:-1]+(binedges[1]-binedges[0])/2.0

    nums = np.zeros(nbins)
    n_succ = np.zeros(nbins)
    fracs = np.zeros(nbins)-1                       
                           
    for ibin in range(nbins):
 
        ind_curr = (np.nonzero((x >= binedges[ibin]) & (x < binedges[ibin+1])))[0]
        nums[ibin] = len(ind_curr)
            
        if len(ind_curr) > 0:
            if mode == 'above':
                ind_above = (np.nonzero(y[ind_curr] > thresh))[0]   
            elif mode == 'at':
                ind_above = (np.nonzero(y[ind_curr] == thresh))[0]
            elif mode == 'below':
                ind_above = (np.nonzero(y[ind_curr] < thresh))[0]
                
            n_succ[ibin] = len(ind_above)            
            fracs[ibin] = len(ind_above)/len(ind_curr)            
            bincen[ibin]=np.median(x[ind_curr])
  
    if uncertainty == True:    
        p_lower = dist.beta.ppf((1-ci)/2.,n_succ+1,nums-n_succ+1)
        p_upper = dist.beta.ppf(1-(1-ci)/2.,n_succ+1,nums-n_succ+1)
    else:
        p_lower = None
        p_upper = None

    ind_plot = (np.nonzero(nums >= dashed_below))[0]    

    if plot:        
        ind_plot_all = (np.nonzero(nums > 0))[0]    
        if dashedline == True and plot_med_line == True:
            plt.plot(bincen[ind_plot_all], fracs[ind_plot_all], linestyle = '--', linewidth = medlinewidth, **kwargs)        
           
        if plot_med_line == True:
            plt.plot(bincen[ind_plot], fracs[ind_plot], linestyle = medlinestyle, linewidth = medlinewidth,label = label, **kwargs)       
    
        if uncertainty:            
            plt.fill_between(bincen[ind_plot], p_lower[ind_plot], p_upper[ind_plot], alpha=alphaunc, **kwargs)
            plt.plot(bincen[ind_plot], p_lower[ind_plot], linewidth=0.5, alpha=0.5, **kwargs)
            plt.plot(bincen[ind_plot], p_upper[ind_plot], linewidth=0.5, alpha=0.5, **kwargs)
     
    if retvals:
        return bincen[ind_plot], fracs[ind_plot], p_lower[ind_plot], p_upper[ind_plot]
    else:
        return bincen[ind_plot], fracs[ind_plot]
        
def plot_average_profile(bincen, y, percent=50, 
                         alphavec = (0.3,0.15,0.7,0.25), 
                         uncertainty = True, scatter = True, mode='median',
                         weight=np.array([]), plot_log = False, refprofile = None,
                         dashed_from = None,
                         plot = True, threshold = None, totweightthresh = None, 
                         black_outline = True, **kwargs):
    
#    set_trace()    
    
    if refprofile == None:
        refprofile = np.zeros_like(bincen)
    
    
    if 'linestyle' in kwargs:
        ls = kwargs['linestyle']
    else:
        ls = 'solid'
    
    # x is a N-vector of the x-points for EACH profile
    # y is a NxM array containing the individual profiles
    
    nums = len(y[0,:])    
    print("nums=", nums)    

    n_full_pt = len(y[:,0])
    ind_plot = np.linspace(0,n_full_pt-1,num=n_full_pt,dtype=int)
    if totweightthresh != None:
        totweights = np.sum(weight,axis=1)
        ind_bad=np.nonzero(totweights < totweightthresh)[0]
        if len(ind_bad) > 0:
            ind_plot = ind_plot[:ind_bad[0]]
    
    if mode == 'median':    
    
        lowlim = 50.0-percent/2.0
        highlim = 50.0+percent/2.0
        percvec = np.array((50, lowlim, highlim, 15.9, 84.1))    


        if threshold == None:
            print("No thresh...")
            percs = np.nanpercentile(y, percvec, axis=1)            
        else:
            print("Evaluating new thresh...")
            percs = np.zeros((len(percvec),len(y[:,0])))
            for iy in range(len(y[:,0])):
                ind_good = (np.nonzero(y[iy,:] >= threshold))[0]

                print("ind_good=", ind_good, " and iy=", iy)                

                if len(ind_good) > 0:
                
                    percarr = y[iy,ind_good]
                    print("Shape of array to percentile is", percarr.shape)
                    percs[:,iy] = np.percentile(percarr, percvec)

        
        print("percs.shape=", percs.shape)
    
        if uncertainty == True:
            lowunc = percs[0,:] - (percs[0,:] - percs[3,:])/np.sqrt(nums)
            highunc = percs[0,:] + (-percs[0,:] + percs[4,:])/np.sqrt(nums)

    if mode == 'mean':

        if len(weight) != len(y):        
            weight = None
            y_masked = y
            is_masked = False            
            
        else:
            print("Suitable weights found...")
            ind_zeroweight = (np.nonzero(weight == 0))
            y[ind_zeroweight] = 0

            y_masked = np.ma.masked_where(weight == 0, y)
            is_masked = True

            
        percs = np.zeros((5,len(bincen)))
        percs[0,:] = np.ma.average(y_masked,weights=weight,axis=1)

        if uncertainty == True:


            stddev = np.ma.MaskedArray.std(y_masked,axis=1)        

            if is_masked:
                nums_per_bin = y_masked.count(axis=1)
            else:
                nums_per_bin = np.zeros(len(stddev))+y_masked.shape[1]

            lowunc = percs[0,:] - stddev/np.sqrt(nums_per_bin)
            highunc = percs[0,:] + stddev/np.sqrt(nums_per_bin)            
                    

    if plot_log:
        percs = np.log10(percs)
 
        if uncertainty:          
            lowunc = np.log10(lowunc)
            ind_badlowunc = np.nonzero(lowunc < -1e10)[0]
            if is_masked:
                ind_badlowunc = np.nonzero(lowunc.mask)[0]
            lowunc[ind_badlowunc] = -10
            highunc = np.log10(highunc)

            print(lowunc)

    if plot == True:

        if mode == 'median':
                
            if uncertainty == True:            
                plt.fill_between(bincen[ind_plot], lowunc[ind_plot]-refprofile[ind_plot], highunc[ind_plot]-refprofile[ind_plot], alpha = alphavec[0], **kwargs)
                plt.plot(bincen[ind_plot], lowunc[ind_plot]-refprofile[ind_plot], linewidth=0.5, alpha=alphavec[2], **kwargs)
                plt.plot(bincen[ind_plot], highunc[ind_plot]-refprofile[ind_plot], linewidth=0.5, alpha=alphavec[2], **kwargs)
                
            if scatter == True:        
                plt.fill_between(bincen[ind_plot], percs[1,ind_plot]-refprofile[ind_plot], percs[2,ind_plot]-refprofile[ind_plot], alpha = alphavec[1], **kwargs)
                plt.plot(bincen[ind_plot], percs[1,ind_plot]-refprofile[ind_plot], linewidth=0.5, alpha=alphavec[3], **kwargs)
                plt.plot(bincen[ind_plot], percs[2,ind_plot]-refprofile[ind_plot], linewidth=0.5, alpha=alphavec[3], **kwargs)


        # Added this bit 17 JAN 16:
        if mode == 'mean':
            if uncertainty == True:
                plt.fill_between(bincen[ind_plot], lowunc[ind_plot]-refprofile[ind_plot], highunc[ind_plot]-refprofile[ind_plot], alpha = alphavec[0], **kwargs)
                plt.plot(bincen[ind_plot], lowunc[ind_plot]-refprofile[ind_plot], linewidth=0.5, alpha=alphavec[2], **kwargs)
                plt.plot(bincen[ind_plot], highunc[ind_plot]-refprofile[ind_plot], linewidth=0.5, alpha=alphavec[2], **kwargs)

       
        if dashed_from != None: 
            plt.plot(bincen[:dashed_from], percs[0,:dashed_from]-refprofile[:dashed_from], linewidth=2.5,color='black', linestyle = ls)
            plt.plot(bincen[:dashed_from], percs[0,:dashed_from]-refprofile[:dashed_from], linewidth=1.8, **kwargs)

            plt.plot(bincen[dashed_from:], percs[0,dashed_from:]-refprofile[dashed_from:], linewidth=2.5,color='black', linestyle = 'dashed')
            plt.plot(bincen[dashed_from:], percs[0,dashed_from:]-refprofile[dashed_from:], linewidth=1.8, linestyle='dashed', **kwargs)

        else:
            if black_outline:
                plt.plot(bincen[ind_plot], percs[0,ind_plot]-refprofile[ind_plot], linewidth=2.5,color='black', linestyle = '-')
            plt.plot(bincen[ind_plot], percs[0,ind_plot]-refprofile[ind_plot], linewidth=1.8, **kwargs)
        
    return percs
    
    
def extract_percentiles(x,y,nbins=10, xrange=None,percrange=[25.0,50.0]):
        
    percrange = np.array(percrange)        
        
    if xrange is None:
        xmin = np.min(x)
        xmax = np.max(x)
    else:
        xmin = xrange[0]
        xmax = xrange[1]

    binedges = np.linspace(xmin, xmax, num=nbins+1, endpoint=True)

    indices = np.array([],dtype=int)
                        
    for ibin in range(nbins):
 
        ind_curr = (np.nonzero((x >= binedges[ibin]) & 
                               (x < binedges[ibin+1])))[0]
            
        if len(ind_curr) > 0:
            
            suby = copy.copy(y[ind_curr])    
            sortinds = np.argsort(suby)
            nums = len(suby)
            minind=np.int(np.rint(percrange[0]/100.0*nums))
            maxind=np.int(np.rint(percrange[1]/100.0*nums))            

            sel_subind = sortinds[minind:maxind]
            sel_ind = ind_curr[sel_subind]
            
            indices = np.append(indices,sel_ind)
            
    return indices            