from scipy.special import hyp2f1
from scipy.special import gamma as g
from scipy.optimize import curve_fit
from numba import vectorize, jit
import numpy as np
import matplotlib.pyplot as plt

@vectorize('float64(float64,float64,int64)')
def EFF_Mr(r, gamma, dim=3):
    if r==0.0: return 0.0
    if r==np.inf: return 1.0
    elif dim==3: return 1 - (2*r**(2 - gamma)*g((1 + gamma)/2.)*hyp2f1((-2 + gamma)/2.,(1 + gamma)/2.,gamma/2.,-r**(-2)))/np.sqrt(np.pi)/g(gamma/2)
    else: return 1. - (1 + r**2.)**(1-gamma/2)

#@jit
def Rebin(data, bins, nmin=8):
    counts = np.histogram(data, bins)[0]
    mask = np.ones_like(bins, dtype=np.bool)
    #i = 0
    while np.any(counts<10):
        index = np.where(counts<10)[0][0]
        if index==0:
            mask[1] = False
        elif index==len(counts)-1:
            mask[-2] = False
        elif counts[index+1] > counts[index-1]:
            mask[index] = False
        elif counts[index+1] < counts[index-1]:
            mask[index+1] = False
        else:
            mask[index] = False
            mask[index+1] = False
        if mask.sum() < 5: return bins
        bins = bins[mask]
        counts = np.histogram(data, bins)[0]
        mask = np.ones_like(bins, dtype=np.bool)
    return bins

def EFF_fit(m, x, phi, h, dim=3, fit_min=256):
    """Returns the fit parameters to the EFF model"""
    if len(m) < fit_min: return np.repeat(np.nan,3), np.repeat(np.nan,3) 
    center = np.average(x, axis=0, weights=m**3 * h**-3)
    nfits = (1 if dim==3 else 3)

    paramlist = []
    errorlist = []
    for i in range(nfits):
#        try:
        if dim==3:
            xproj = x-center
        else:
            if i==0:
                xproj = np.c_[x[:,0]-center[0],x[:,1]-center[1]]
            elif i==1:
                xproj = np.c_[x[:,0]-center[0],x[:,2]-center[2]]
            else:
                xproj = np.c_[x[:,1]-center[1],x[:,2]-center[2]]
        r = np.sum(xproj**2, axis=1)**0.5
        r50 = np.median(r)
        logrmin = np.log10(np.sort(r)[4])
        logrmax = np.log10(np.percentile(r,95))
        nbins = 16
        rbins = np.logspace(logrmin,logrmax,nbins+1)
        rbins[0] = 0.

        count = np.histogram(r, bins=rbins)[0]
        #while np.any(count==0):
        #    nbins /= 2
        #    if nbins < 4: break
        #    rbins = np.logspace(logrmin,logrmax,nbins+1)
        #    rbins[0] = 0.
        #    count = np.histogram(r, rbins)[0]
        #if len(rbins) > 5:
        rbins = Rebin(r, rbins)

        #for rad in rbins[1:]:
        #    phi = np.linspace(0,2*np.pi,1000)
        #    plt.plot(rad*np.cos(phi), rad*np.sin(phi),color='black')
        #plt.scatter((x-center)[:,0],(x-center)[:,1],s=0.1); plt.axes().set_aspect('equal'); plt.show()
        mbin = np.histogram(r, rbins, weights=m)[0]
        r_avg = 0.5*(rbins[1:]+rbins[:-1])
        count = np.histogram(r, rbins)[0]
        #if dim==2:
        #    density = mbin/np.diff(rbins**2)/np.pi
        delta_M = lambda r, *params: np.exp(params[0])*np.diff(EFF_Mr(rbins/np.exp(params[1]), params[2],dim))
        #    density_model = lambda r, *params: params[0] * (1+(r/np.exp(params[1]))**2.)**(-params[2]/2)
        #else:
        #    density = mbin/np.diff(4*np.pi*rbins**3 / 3)
        #    density_model = lambda r, *params: params[0] * (1+(r/np.exp(params[1]))**2.)**(-(params[2]+1)/2)
        #params, pcov = curve_fit(density_model, r_avg, density, (
        params, pcov = curve_fit(delta_M, rbins[1:], mbin, (np.log(m.sum()), np.log(r50/10), 2.5), maxfev=10**4, sigma=mbin/count**0.5)
        #plt.loglog(r_avg, mbin/np.diff(rbins**dim), r_avg, delta_M(0.,*params)/np.diff(rbins**dim)); plt.show()
        params[1] = np.exp(params[1])
        params[0] = np.exp(params[0])
        errors = np.diag(pcov)**0.5
        errors[1] *= params[1]
        errors[0] *= params[0]
#        except:
#            params, errors = np.repeat(np.nan,3), np.repeat(np.nan,3)
        paramlist.append(params)
        errorlist.append(errors)
    if nfits==1: return paramlist[0], errorlist[0]
    else:
        return np.average(paramlist,axis=0, weights=np.array(errorlist)**-2.), np.max([np.max(errorlist,axis=0), np.std(paramlist,axis=0)],axis=0)
