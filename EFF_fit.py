from scipy.special import hyp2f1
from scipy.special import gamma as g
from scipy.optimize import curve_fit
from numba import vectorize, jit
import numpy as np
import matplotlib.pyplot as plt
from meshoid import meshoid

@vectorize('float64(float64,float64,int64)')
def EFF_Mr(r, gamma, dim=3):
    if r==0.0: return 0.0
    if dim==2:
        return 2*np.pi/(gamma-2) * (1. - (1 + r**2.)**(1-gamma/2))
    else:
        return (2*np.pi*(1 + r**2)**((-1 - gamma)/2.)*(1 - r**2*(-4 + gamma*(4 + gamma*r**2)) - gamma*r**2*(-1 + (-2 + gamma)*r**2)*(3 + gamma*r**2)*hyp2f1(1,1 - gamma/2.,1.5,-r**2) + (-1 + (-2 + gamma)*r**2*(2 + gamma*r**2))*hyp2f1(1,-gamma/2.,0.5,-r**2)))/((-2 + gamma)*gamma*r)

#    if r==np.inf: return 1.0
#    elif dim==3:
#        if gamma>2:
#            return 1 - (2*r**(2 - gamma)*g((1 + gamma)/2.)*hyp2f1((-2 + gamma)/2.,(1 + gamma)/2.,gamma/2.,-r**(-2)))/np.sqrt(np.pi)/g(gamma/2)
#        else:
#            return ((1 + r**2)**(0.5 - gamma/2.)*g((1 + gamma)/2.)*(1 + gamma*r**2 - hyp2f1(1,-gamma/2.,0.5,-r**2)))/(np.sqrt(np.pi)*r*g(1 + gamma/2.))
#    else: 

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

def EFF_fit(m, x, phi, h=None, dim=2, fit_min=100, path=None, nfits=3):
    """Returns the fit parameters to the EFF model"""
    if len(m) < fit_min: return np.repeat(np.nan,3), np.repeat(np.nan,3), np.nan, np.nan 

    center = np.average(x, weights=np.abs(phi)**2,axis=0)
    
    nfits = (1 if dim==3 else nfits)

    paramlist = []
    errorlist = []
    rChiSqr = []
    delta_m = []

    for i in range(nfits):
        try:
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
            if h is None:
                h = meshoid(x,m).SmoothingLength()

            logrmin = np.log10(h.min()) #np.log10(np.sort(r)[4])
            logrmax = np.log10(r50)
#            nbins = 16
            rbins = np.logspace(logrmin,np.log10(np.percentile(r,95)),15) #np.log10(r50*10),20)
#            rbins = np.logspace(logrmin,logrmax,nbins+1)
            rbins[0] = 0.

#            count = np.histogram(r, bins=rbins)[0]

#            rbins = Rebin(r, rbins)
            if dim==2:
                dvol = np.pi*np.diff(rbins**2)
            else:
                dvol = 4*np.pi/3 * np.diff(rbins**3)

            mbin = np.histogram(r, rbins, weights=m)[0]
            r_avg = float(dim)/(dim+1)*np.diff(rbins**(dim+1))/np.diff(rbins**dim)  #(0.5*(rbins[1:]**(dim-1)+rbins[:-1]**(dim-1)))**(1./(dim-1))

            count = np.histogram(r, rbins)[0]
            errors = mbin/np.clip(count,1,1e100)**0.5
            errors[count==0] = np.inf

            delta_M = lambda r, *params: np.exp(params[0])*np.exp(2*params[1])*np.diff(EFF_Mr(rbins/np.exp(params[1]), params[2],dim))

            a0 = r50/10
            rho0 =  mbin[mbin>0][0]/dvol[mbin>0][0]

#            for gamma0 in 2+np.logspace(-1,1,10):
            #    try:
            params, pcov = curve_fit(delta_M, rbins[1:], mbin, (np.log(rho0), np.log(a0), 2.1), maxfev=10**4, sigma=errors)
 #                   if not (np.any(params==np.nan) or np.abs(pcov[2,2]**0.5/params[2])>1): break
#                except:
#                    continue
#            if np.any(params==np.nan): raise("Could not achieve a good fit.")


            rorder = r.argsort()
            Mr = m[rorder].cumsum()
            frac = np.arange(1./len(m)/2,1-1./len(m)/2, 1./len(m))

            dm_fit = delta_M(rbins,*params)
#            print dm_fit, mbin
#            print (dm_fit/dvol)[0], np.exp(params[0])
            if len(x) > 10**4:
                plt.loglog(r_avg,mbin/dvol, r_avg, dm_fit/dvol); plt.show()
            sigmaN_expected = (np.clip(mbin,m.mean(),1e100)/m.mean())**-0.5
            sigmam_expected = sigmaN_expected * mbin

            if path:
                print(count.shape, r_avg.shape, path+"profile.dat")
                np.savetxt(path + "profile.dat", np.c_[r_avg, mbin/dvol, delta_M(rbins[1:],*params)/dvol, count])
                
            params[1] = np.exp(params[1])
            params[0] = np.exp(params[0])

            errors = np.diag(pcov)**0.5
            errors[1] *= params[1]
            errors[0] *= params[0]

            paramlist.append(params)
            errorlist.append(errors)

            rChiSqr.append(np.sum((dm_fit[mbin>0]-mbin[mbin>0])**2/sigmam_expected[mbin>0]**2)/(len(mbin)-3))

            delta_m.append(np.average(((dm_fit-mbin)[mbin>0]/mbin[mbin>0])**2, weights=mbin[mbin>0])**0.5)

        
        except Exception as e:
            print(e)
            paramlist.append(np.repeat(np.nan,3))
            errorlist.append(np.repeat(np.nan, 3))
            rChiSqr.append(np.nan)
            delta_m.append(np.nan)
            #, errors = np.repeat(np.nan,3), np.repeat(np.nan,3)
        
        
    if nfits==1: return paramlist[0], errorlist[0], delta_m[0], rChiSqr[0]
    else: return np.average(paramlist,axis=0, weights=np.array(errorlist)**-2.), np.max([np.max(errorlist,axis=0), np.std(paramlist,axis=0)],axis=0), np.max(delta_m), np.max(rChiSqr)
