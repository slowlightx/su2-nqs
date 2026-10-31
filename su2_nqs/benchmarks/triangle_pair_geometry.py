"""Explicit periodic geometry for two oriented elementary triangular probes.

No state, ansatz, CG data, or Hamiltonian action is constructed here. Physical
row-major sites use the120-degree metric dr²+dc²-dr*dc. Frozen six-spin probe
indices are A=(0,3,4), B=(1,5,2); order is part of the physical definition.
"""
from __future__ import annotations

from itertools import combinations, product
from numbers import Integral
from typing import Sequence

import numpy as np


def _length(length: int) -> int:
    if isinstance(length,(bool,np.bool_)) or not isinstance(length,Integral) or length<6:
        raise ValueError('integer triangular torus length>=6 required')
    return int(length)


def _site(site: int,length: int) -> int:
    if isinstance(site,(bool,np.bool_)) or not isinstance(site,Integral) or not 0<=site<length**2:
        raise ValueError('integer physical row-major site outside torus')
    return int(site)


def _norm(delta: Sequence[int]) -> int:
    a,b=map(int,delta)
    return a*a+b*b-a*b


def _shortest_numerators(delta: np.ndarray,period: int) -> tuple[int,list[list[int]]]:
    candidates=[delta+period*np.asarray(shift) for shift in product((-1,0,1),repeat=2)]
    norms=[_norm(d) for d in candidates];best=min(norms)
    return best,[d.tolist() for d,n in zip(candidates,norms) if n==best]


def periodic_triangle(triangle: Sequence[int],length: int=6) -> dict:
    """Validate an elementary J1 triangle and return a consistent local lift.

    Orientation+1 is CCW in e_r=(1,0),e_c=(-1/2,sqrt3/2). ``flavor`` is the
    translation-invariant centroid residue mod3, distinguishing up/down faces.
    """
    length=_length(length)
    triangle=tuple(_site(s,length) for s in triangle)
    if len(triangle)!=3 or len(set(triangle))!=3:
        raise ValueError('three distinct ordered physical sites required')
    anchor=np.asarray(divmod(triangle[0],length));lift=[anchor]
    for site in triangle[1:]:
        delta=np.asarray(divmod(site,length))-anchor
        norm,images=_shortest_numerators(delta,length)
        if norm!=1 or len(images)!=1:
            raise ValueError('not an elementary nonwinding J1 triangle')
        lift.append(anchor+images[0])
    if _norm(lift[2]-lift[1])!=1:
        raise ValueError('triangle has a non-J1 third edge')
    a,b=lift[1]-lift[0],lift[2]-lift[0]
    determinant=int(a[0]*b[1]-a[1]*b[0])
    center=np.sum(lift,axis=0)
    return dict(sites=list(triangle),lifted_coordinates=[p.tolist() for p in lift],
        orientation=1 if determinant>0 else -1,
        area=abs(determinant)*np.sqrt(3)/4,
        centroid_numerator_over3=center.tolist(),flavor='_'.join(map(str,center%3)))


def ccw_triangle(triangle: Sequence[int],length: int=6) -> tuple[int,int,int]:
    """Return a deterministic CCW order, keeping the first physical site."""
    data=periodic_triangle(triangle,length)
    t=tuple(data['sites'])
    return t if data['orientation']==1 else (t[0],t[2],t[1])


def triangle_pair_geometry(a: Sequence[int],b: Sequence[int],length: int=6) -> dict:
    """Distances, face flavors, overlap and every geometric J1/J2 edge.

    Center differences are exact integer numerators over3; squared distances
    have denominator9. Equal shortest torus images are all retained.
    """
    length=_length(length);aa=periodic_triangle(a,length);bb=periodic_triangle(b,length)
    delta=np.asarray(bb['centroid_numerator_over3'])-aa['centroid_numerator_over3']
    n2,displacements=_shortest_numerators(delta,3*length)
    aset,bset=set(aa['sites']),set(bb['sites']);union=sorted(aset|bset)
    edges={1:[],3:[]}
    for i,j in combinations(union,2):
        d=np.asarray(divmod(j,length))-np.asarray(divmod(i,length))
        n,_=_shortest_numerators(d,length)
        if n in edges:
            edges[n].append(dict(sites=[i,j],cross_triangle=(i in aset and j in bset) or (j in aset and i in bset)))
    return dict(A=aa,B=bb,overlap=sorted(aset&bset),disjoint=not(aset&bset),
        same_triangle_flavor=aa['flavor']==bb['flavor'],
        shortest_center_displacement_numerators_over3=displacements,
        shortest_center_distance_squared_numerator_over9=n2,
        shortest_center_distance_squared=n2/9,shortest_center_distance=float(np.sqrt(n2/9)),
        geometric_J1_edges=edges[1],geometric_J2_edges=edges[3],
        warning='J2 geometric edges are Hamiltonian couplings only when J2 is nonzero; face flavor and displacement direction must be controlled independently of distance.')


def frozen_six_site_order(a: Sequence[int],b: Sequence[int],length: int=6) -> tuple[int,...]:
    """Map ordered triangles to fixed A=(0,3,4),B=(1,5,2) operator positions."""
    geometry=triangle_pair_geometry(a,b,length)
    if not geometry['disjoint']:
        raise ValueError('overlapping triangles cannot use a disjoint six-spin operator')
    aa,bb=geometry['A']['sites'],geometry['B']['sites']
    return aa[0],bb[0],bb[2],aa[1],aa[2],bb[1]


def disjoint_elementary_triangle_pairs(sites: Sequence[int],length: int=6) -> tuple[dict,...]:
    """Enumerate candidates, never infer a distance sequence from region names."""
    length=_length(length);sites=tuple(_site(s,length) for s in sites)
    if len(set(sites))!=len(sites):
        raise ValueError('unique region sites required')
    triangles=[]
    for triple in combinations(sorted(sites),3):
        try: triangles.append(ccw_triangle(triple,length))
        except ValueError: continue
    return tuple(triangle_pair_geometry(a,b,length) for a,b in combinations(triangles,2) if not(set(a)&set(b)))
