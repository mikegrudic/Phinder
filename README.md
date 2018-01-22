# Phinder: robust bound star cluster finding via potential well grouping
```python
Usage: Phinder.py <files> ... [options]

Options:                                                                       
   -h --help                  Show this screen.
   --softening=<L>            Gravitational softening used if adaptive softenings not found [default: 0.1]
   --ptype=<N>                GIZMO particle type to analyze [default: 4]
   --G=<G>                    Gravitational constant to use; should be consistent with what was used in the simulation. [default: 1.0]
   --boxsize=<L>              Box size of the simulation; for neighbour-search purposes. [default: None]
   --cluster_ngb=<N>          Length of particle's neighbour list. [default: 32]
   --min_cluster_size=<N>     Minimum number of particles in cluster. [default: 32]
   --brute_force_N=<N>        Maximum number of particles in a cluster before we compute the potential in the spherically-symmetric approximation. [default: 100000]
   --fuzz=<L>                 Randomly perturb particle positions by this small fraction to avoid problems with particles at the same position in 32bit floating point precision data [default: 0]
   --fits=<N>                 Fit clusters to EFF profile: 0 if no, 2 if fitting surface density, 3 if fitting 3D density. [default: 0]
'''