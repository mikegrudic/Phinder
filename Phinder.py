#!/usr/bin/env python
"""                                                                            
Phinder: robust bound star cluster finding via potential well grouping

Usage: Phinder.py <files> ... [options]

Options:                                                                       
   -h --help                  Show this screen.
   --softening=<L>            Gravitational softening used if adaptive softenings not found [default: 0.1]
   --ptype=<N>                GIZMO particle type to analyze [default: 4]
   --G=<G>                    Gravitational constant to use; should be consistent with what was used in the simulation. [default: 4.301e4]
   --boxsize=<L>              Box size of the simulation; for neighbour-search purposes. [default: None]
   --cluster_ngb=<N>          Length of particle's neighbour list. [default: 32]
   --min_cluster_size=<N>     Minimum number of particles in cluster. [default: 32]
   --brute_force_N=<N>        Maximum number of particles in a cluster before we compute the potential in the spherically-symmetric approximation. [default: 100000]
   --recompute_potential      Use this if you want to compute the potential on the fly rather than use the snapshot's [default: False]
   --fuzz=<L>                 Randomly perturb particle positions by this small fraction to avoid problems with particles at the same position in 32bit floating point precision data [default: 0]
   --fits=<N>                 Fit clusters to EFF profile: 0 if no, 2 if fitting surface density, 3 if fitting 3D density. [default: 0]
"""

import h5py
from numba import jit, vectorize
import numpy as np
from sys import argv
from scipy import integrate
import matplotlib.pyplot as plt
import meshoid
from docopt import docopt
from collections import OrderedDict
from EFF_fit import EFF_fit
import pykdgrav
from os import path
    
@jit
def ComputePotential(x, m, h, G=1.):
    """Computes the gravitational potential via brute force
       x - (N,3) array of positions
       m - (N,) array of masses
       h - (N,) array of softening lengths """
    N = len(m)
    phi = np.zeros_like(m)
    for i in xrange(N):
        if h[i] > 0: phi[i] += -G*m[i]/h[i]
        for j in xrange(i+1,N):
            rijSqr = (x[i,0]-x[j,0])**2 + (x[i,1]-x[j,1])**2 + (x[i,2]-x[j,2])**2
            rij = np.sqrt(rijSqr)
            if rij > h[j]:
                phi[i] += -G*m[j]/rij
            else:
                phi[i] += -G*m[j]/np.sqrt(rijSqr + h[j]**2.)
            if rij > h[i]:
                phi[j] += -G*m[i]/rij
            else:
                phi[j] += -G*m[i]/np.sqrt(rijSqr + h[i]**2.)
    return phi

@jit
def FindOwners(ngb, phi,r):
    """Determines the 'owner' of each particle given the neighbour list, gravitational potential, and neighbor distances"""
    i = 0
    owners = -np.ones(len(phi), dtype=np.int32)
    
    for i in xrange(len(phi)):
        Owner(i, ngb, owners, phi, r)
    return owners

@jit
def Owner(i, ngb, owners, phi,r):
    """Called by FindOwners and recursive when doing the neighbor walk for a particle"""
    if owners[i] > -1:
        return owners[i]
    elif ngb[i][phi[ngb[i]].argmin()] == i:
        owners[i] = i
        return i
    else:
        owners[i] = Owner(ngb[i][phi[ngb[i]].argmin()], ngb, owners, phi,r)
        return owners[i]

def SaveArrayDict(path, arrdict):
    """Takes a dictionary of numpy arrays with names as the keys and saves them in an ASCII file with a descriptive header"""
    header = ""
    offset = 0
    
    for i, k in enumerate(arrdict.keys()):
        if type(arrdict[k])==list: arrdict[k] = np.array(arrdict[k])
        if len(arrdict[k].shape) == 1:
            header += "(%d) "%offset + k + "\n"
            offset += 1
        else:
            header += "(%d-%d) "%(offset, offset+arrdict[k].shape[1]-1) + k + "\n"
            offset += arrdict[k].shape[1]
            
    data = np.column_stack([b for b in arrdict.values()])
    data = data[(-data[:,0]).argsort()] 
    np.savetxt(path, data, header=header,  fmt='%10.5g', delimiter='\t')

#@jit
#def delta_M(rbins, params):
#    result = params[0]*np.diff(EFF_Mr(r/params[1], params[2],dim)) - mbin
    


def ComputeClusters(filename, options):
    brute_force_N = int(float(options["--brute_force_N"]) + 0.5)
    cluster_ngb = int(float(options["--cluster_ngb"]) + 0.5)
    min_cluster_size = int(float(options["--cluster_ngb"]) + 0.5)
    softening = float(options["--softening"])
    G = float(options["--G"])
    boxsize = options["--boxsize"]
    ptype = "PartType"+ options["--ptype"]
    recompute_potential = options["--recompute_potential"]
    if boxsize != "None":
        boxsize = float(boxsize)
    else:
        boxsize = None
    fuzz = float(options["--fuzz"])
    fits = int(float(options["--fits"])+0.5)

    if path.isfile(filename):
        F = h5py.File(filename)
    else:
        print("Could not find "+filename)
        return
    if not ptype in F.keys():
        print("Particles of desired type not found!")

    m = np.array(F[ptype]["Masses"])
    criteria = np.ones(len(m),dtype=np.bool)
    m = m[criteria]

    if len(m) < 32:
        print("Not enough particles for meaningful cluster analysis!")
        return
    
    x = np.array(F[ptype]["Coordinates"])[criteria]
    if ptype=="PartType0":
        u = np.array(F[ptype]["InternalEnergy"])
        rho = np.array(F[ptype]["Density"])
        criteria *= (u < 10.)*(rho*404 > 10.) # only look at cold dense gas (<100K, >10cm^-3)
        print(criteria.sum())
        m = m[criteria]
        x = x[criteria]
        recompute_potential = True


    if fuzz: x += np.random.normal(size=x.shape)*x.std()*fuzz
    if "Potential" in F[ptype].keys() and not recompute_potential:
        phi = np.array(F[ptype]["Potential"])[criteria]
    else:
        phi = pykdgrav.Potential(x,m,G=G,theta=1.)
    if "AGS-Softening" in F[ptype].keys():
        h_ags = np.array(F[ptype]["AGS-Softening"])[criteria]
    elif "SmoothingLength" in F[ptype].keys():
        h_ags = np.array(F[ptype]["SmoothingLength"])[criteria]
    else:
        h_ags = softening*np.ones_like(m)
    hmin = h_ags.min()
        
    v = np.array(F[ptype]["Velocities"])[criteria]
    
    print("Finding neighbors...")
    mm = meshoid.meshoid(x, m, des_ngb=cluster_ngb, boxsize=boxsize)
    h = mm.h

    ngbdist, ngb = mm.ngbdist, mm.ngb #tree.query(x, cluster_ngb)
    print("Done!")

    owners = -np.ones(len(phi), dtype=np.int32)
    owners = FindOwners(ngb, phi,ngbdist)
    
    clusters = OrderedDict()
    for i, o in enumerate(owners):
        if not o in clusters.keys():
            clusters[o] = []
        clusters[o].append(i)

    # have to merge any spurious double-clusters
    clusters_merged = {}
    for c in clusters.keys():
#        dx = np.sum((x[clusters[c]] - x[c])**2, axis=1)**0.5
        r1s = 4*hmin
        dxc = np.sum((x[clusters.keys()] - x[c])**2, axis=1)**0.5
        # is there are no clusters within the 10% radius, simply copy the original cluster. Otherwise, merge the clusters in proximity
        if not np.any(np.sort(dxc)[1:] < r1s):
            clusters_merged[c] = clusters[c]
        else:
            #figure out which one has the lowest potential out of the clusters within the 10% radius. Merge all others into that one.
            within_r = np.array(clusters.keys())[dxc < r1s]
            parent = within_r[phi[within_r].argmin()]
            cluster = []
            for i in within_r:
                cluster += clusters[i]
                clusters_merged[parent] = cluster #sum([clusters[i] for i in within_r])

    clusters = clusters_merged

    # This ends the assignment of clusters to potential wells; the dictionary clusters contains the indices if that cluster's particles

    # Now we determine the bound subsets of the clusters

#    csize = [len(c) for c in clusters.values()]
    rejects = []

    bound_data = OrderedDict()
    bound_data["Mass"] = []
    bound_data["Center"] = []
    bound_data["HalfMassRadius"] = []
    bound_data["NumParticles"] = []
    if fits:
        bound_data["EFF_gamma"] = []
        bound_data["EFF_gamma_error"] = []
        bound_data["EFF_a"] = []
        bound_data["EFF_a_error"] = []
        bound_data["EFF_DeltaM"] = []
        bound_data["EFF_rChiSqr"] = []
    unbound_data = OrderedDict()
    unbound_data["Mass"] = []
    unbound_data["Center"] = []
    unbound_data["HalfMassRadius"] = []
    unbound_data["NumParticles"] = []
    unbound_data["BoundFraction"] = []

    n = filename.split("snapshot_")[1].split(".")[0]

    Fout = h5py.File(filename.split("snapshot")[0] + "Clusters_%s.hdf5"%n, 'w')

    print("Selecting bound subsets...")

    rejects = []
    bound_clusters = []    
    for k,c in clusters.items():
        if len(c) < min_cluster_size:
            rejects.append(c)
            continue
        c = np.array(c)
        xc = x[c]
        phic = phi[c]
        r = np.sum((xc - xc[phic.argmin()])**2,axis=1)**0.5
        rorder = r.argsort()
        c = c[rorder]
        xc = xc[rorder]
        phic = phic[rorder]
        vc = v[c]
        hc = h[c]
        mc = m[c]
        r = r[rorder]

        unbound_data["Mass"].append(mc.sum())
        unbound_data["NumParticles"].append(len(mc))
        unbound_data["Center"].append(xc[phic.argmin()])
        unbound_data["HalfMassRadius"].append(np.median(r))

        Mr = mc.cumsum()
        if len(c) < brute_force_N:
            phi2 = ComputePotential(xc, mc, h_ags[c]/2.8, G) # direct summation
        else:
            phi2 = pykdgrav.Potential(np.float64(xc), np.float64(mc), G=G)
            #phi2 = G*integrate.cumtrapz(Mr[::-1]/(r[::-1]**2 + (h_ags[c]/2.8)**2), x=r[::-1], initial=0.0)[::-1] - G*mc.sum()/r[-1] # spherical symmetry approximation


        rho = mc/(4*np.pi*hc**3/3)
        v_cluster = np.average(vc,axis=0,weights=mc*np.abs(phi2))#rho**2)
#        x_cluster = np.average(xc,axis=0,weights=mc*rho**2)
        vSqr = np.sum((vc - v_cluster)**2,axis=1)

        bound = 0.5*vSqr + phi2 < 0


        unbound_data["BoundFraction"].append(float(bound.sum())/len(bound))

        rejects.append(c[np.invert(bound)])

        if bound.sum() > min_cluster_size:
            bound_clusters.append(c[bound])
            c = c[bound]
            bound_data["Mass"].append(mc[bound].sum())
            bound_data["NumParticles"].append(len(mc[bound]))
            bound_data["Center"].append(np.average(xc[bound],weights=phi2[bound]**2, axis=0))
            bound_data["HalfMassRadius"].append(np.median(r[bound]))
            if fits:
                #print EFF_fit(mc[bound], xc[bound], phic[bound], hc[bound], dim=fits)
                EFF_params, EFF_errors, EFF_dm, EFF_rChiSqr = EFF_fit(mc[bound], xc[bound], phi2[bound], hc[bound], dim=fits)#, path=filename.split("snapshot")[0])
                bound_data["EFF_gamma"].append(EFF_params[2])
                bound_data["EFF_gamma_error"].append(EFF_errors[2])
                bound_data["EFF_a"].append(EFF_params[1])
                bound_data["EFF_a_error"].append(EFF_errors[1])
                bound_data["EFF_DeltaM"].append(EFF_dm)
                bound_data["EFF_rChiSqr"].append(EFF_rChiSqr)

    print("Done grouping bound clusters!")
#                print "Reduced chi^2: %g Relative mass error: %g"%(EFF_rChiSqr,EFF_dm/mc[bound].sum())
#    cluster_masses = np.array(bound_data["Mass"])
    bound_clusters = np.array(bound_clusters)[np.array(bound_data["Mass"]).argsort()[::-1]]
    
    # write to Clusters_xxx.hdf5
    for i, c in enumerate(bound_clusters):
        cluster_id = "Cluster"+ ("%d"%i).zfill(int(np.log10(len(bound_clusters))+1))
        N = len(c)
        Fout.create_group(cluster_id)
        for k in F[ptype].keys():
            Fout[cluster_id].create_dataset(k, data=np.array(F[ptype][k])[criteria][c])
        
    Fout.close()
    F.close()
    
    #now save the ascii data files
    SaveArrayDict(filename.split("snapshot")[0] + "bound_%s.dat"%n, bound_data)
    SaveArrayDict(filename.split("snapshot")[0] + "unbound_%s.dat"%n, unbound_data)

    
    
def main():
    options = docopt(__doc__)
    for f in options["<files>"]:
        print(f)
        ComputeClusters(f, options)

if __name__ == "__main__": main()
