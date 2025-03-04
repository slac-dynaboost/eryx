import numpy as np
import glob
import re
import os
from .pdb import AtomicModel
from .map_utils import *
from .scatter import structure_factors
from eryx.autotest.debug import debug

def natural_sort(l): 
    """
    Natural sort items in list. Helper function for guinier_reconstruct.
    
    Parameters
    ----------
    l : list of str
        list of strings to natural sort
    
    Returns
    -------
    naturally-sorted list of str
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def guinier_reconstruct(ensemble_dir, n_grid_points, res_mask, n_asu, weights=None):
    """
    Reconstruct a map from an ensemble of complex structure factors
    using Guinier's equation. The ensemble directory should contain
    a single file per asu / ensemble member, which are incoherently
    summed assuming that values at unmasked grid points were saved.
    
    Parameters
    ----------
    ensemble_dir : str
        path to directory containing ensemble's structure factors
    n_grid_points : int
        number of q-grid points, i.e. flattened map size
    res_mask : numpy.ndarray, shape (n_grid_points,)
        boolean resolution mask
    n_asu : int
        number of asymmetric units
    weights : numpy.ndarray, shape (n_conformations,)
        weights associated with each state; uniform if not provided
    """
    fnames = glob.glob(os.path.join(ensemble_dir, "*npy"))
    fnames = natural_sort(fnames)
    
    if weights is None:
        weights = np.ones(int(len(fnames) / n_asu)) / (len(fnames) / n_asu)
    
    Id = np.zeros(n_grid_points)
    for asu in range(n_asu):
        fnames_asu = fnames[asu::n_asu]
        fc = np.zeros(n_grid_points, dtype=complex)
        fc_square = np.zeros(n_grid_points)
        
        for i,fname in enumerate(fnames_asu):
            print(fname)
            A = np.load(fname)
            fc[res_mask] += A * weights[i]
            fc_square[res_mask] += np.square(np.abs(A)) * weights[i]
        Id += fc_square - np.square(np.abs(fc))
    
    Id[~res_mask] = np.nan
    return Id

@debug
def compute_crystal_transform(pdb_path, hsampling, ksampling, lsampling, U=None, expand_p1=True, 
                              res_limit=0, batch_size=5000, n_processes=8):
    """
    Compute the crystal transform as the coherent sum of the
    asymmetric units. If expand_p1 is False, it is assumed 
    that the pdb contains asymmetric units as separate frames.
    The crystal transform is only defined at integral Miller 
    indices, so grid points at fractional Miller indices or 
    beyond the resolution limit will be set to zero.
    
    Parameters
    ----------
    pdb_path : str
        path to coordinates file 
    hsampling : tuple, shape (3,)
        (hmin, hmax, oversampling) relative to Miller indices
    ksampling : tuple, shape (3,)
        (kmin, kmax, oversampling) relative to Miller indices
    lsampling : tuple, shape (3,)
        (lmin, lmax, oversampling) relative to Miller indices
    expand_p1 : bool
        if True, expand PDB (asymmetric unit) to unit cell
    U : numpy.ndarray, shape (n_atoms,)
        isotropic displacement parameters, applied to each asymmetric unit
    res_limit : float
        high resolution limit
    batch_size : int
        number of q-vectors to evaluate per batch 
    n_processes : int
        number of processors over which to parallelize the calculation
        
    Returns
    -------
    q_grid : numpy.ndarray, (n_points, 3)
        q-vectors corresponding to flattened intensity map
    I : numpy.ndarray, 3d
        intensity map of the crystal transform
    """
    model = AtomicModel(pdb_path, expand_p1=expand_p1, frame=-1)
    model.flatten_model()
    hkl_grid, map_shape = generate_grid(model.A_inv, 
                                        hsampling,
                                        ksampling, 
                                        lsampling, 
                                        return_hkl=True)
    q_grid = 2*np.pi*np.inner(model.A_inv.T, hkl_grid).T
    mask, res_map = get_resolution_mask(model.cell, hkl_grid, res_limit)
    dq_map = np.around(get_dq_map(model.A_inv, hkl_grid), 5)
    dq_map[~mask] = -1
    
    I = np.zeros(q_grid.shape[0])
    I[dq_map==0] = np.square(np.abs(structure_factors(q_grid[dq_map==0],
                                                      model.xyz, 
                                                      model.ff_a,
                                                      model.ff_b,
                                                      model.ff_c,
                                                      U=U, 
                                                      batch_size=batch_size,
                                                      n_processes=n_processes)))
    return q_grid, I.reshape(map_shape)

@debug
def compute_molecular_transform(pdb_path, hsampling, ksampling, lsampling, U=None, expand_p1=True,
                                expand_friedel=True, res_limit=0, batch_size=10000, n_processes=8):
    """
    Compute the molecular transform as the incoherent sum of the 
    asymmetric units. If expand_p1 is False, the pdb is assumed 
    to contain the asymmetric units as separate frames / models.
    The calculation is accelerated by leveraging symmetry in one
    of two ways, one of which will maintain the input grid extents
    (expand_friedel=False), while the other will output a map that 
    includes the volume of reciprocal space related by Friedel's law.
    If h/k/lsampling are symmetric about (0,0,0), these approaches 
    will yield identical maps. If expand_friedel is False and the
    space group is P1, the simple sum over asus will be performed
    to avoid wasting time on determining symmetry relationships.

    Parameters
    ----------
    pdb_path : str
        path to coordinates file 
    hsampling : tuple, shape (3,)
        (hmin, hmax, oversampling) relative to Miller indices
    ksampling : tuple, shape (3,)
        (kmin, kmax, oversampling) relative to Miller indices
    lsampling : tuple, shape (3,)
        (lmin, lmax, oversampling) relative to Miller indices
    U : numpy.ndarray, shape (n_atoms,)
        isotropic displacement parameters, applied to each asymmetric unit
    expand_p1 : bool
        if True, expand PDB (asymmetric unit) to unit cell
    expand_friedel : bool
        if True, expand to full sphere in reciprocal space
    res_limit : float
        high resolution limit
    batch_size : int
        number of q-vectors to evaluate per batch 
    n_processes : int
        number of processors over which to parallelize the calculation
        
    Returns
    -------
    q_grid : numpy.ndarray, (n_points, 3)
        q-vectors corresponding to flattened intensity map
    I : numpy.ndarray, 3d
        intensity map of the molecular transform
    """
    model = AtomicModel(pdb_path, expand_p1=expand_p1)
    hkl_grid, map_shape = generate_grid(model.A_inv, 
                                        hsampling,
                                        ksampling, 
                                        lsampling, 
                                        return_hkl=True)
    q_grid = 2*np.pi*np.inner(model.A_inv.T, hkl_grid).T
    mask, res_map = get_resolution_mask(model.cell, hkl_grid, res_limit)
    sampling = (hsampling[2], ksampling[2], lsampling[2])

    if model.space_group == 'P 1' and not expand_friedel:
        I = np.zeros(q_grid.shape[0])
        for asu in range(model.xyz.shape[0]):
            I[mask] += np.square(np.abs(structure_factors(q_grid[mask],
                                                          model.xyz[asu],
                                                          model.ff_a[asu], 
                                                          model.ff_b[asu], 
                                                          model.ff_c[asu],
                                                          U=U,
                                                          batch_size=batch_size,
                                                          n_processes=n_processes)))
        I = I.reshape(map_shape)
    else:
        if expand_friedel:
            I = incoherent_sum_real(model, hkl_grid, sampling, U, mask, batch_size, n_processes)
        else:
            I = incoherent_sum_reciprocal(model, hkl_grid, sampling, U, batch_size, n_processes)
            I = I.reshape(map_shape)
            I[~mask.reshape(map_shape)] = 0

    return q_grid, I

@debug
def incoherent_sum_real(model, hkl_grid, sampling, U=None, mask=None, batch_size=10000, n_processes=8):
    """
    Compute the incoherent sum of the scattering from all asus.
    The scattering for the unique reciprocal wedge is computed 
    by summing over all asymmetric units in real space, and then
    using symmetry to extend the calculation to the remainder of
    the map (including the portion of reciprocal space related by
    Friedel's law even if not spanned by the input hkl_grid).
    
    Parameters
    ----------
    model : AtomicModel
        instance of AtomicModel class expanded to p1
    hkl_grid : numpy.ndarray, shape (n_points, 3)
        hkl vectors of map grid points
    sampling : tuple
        sampling frequency along h,k,l axes
    U : numpy.ndarray, shape (n_atoms,)
        isotropic displacement parameters, applied to each asymmetric unit
    batch_size : int
        number of q-vectors to evaluate per batch
    mask : numpy.ndarray, shape (n_points,)
        boolean mask, where True indicates grid points to keep
    n_processes : int
        number of processors over which to parallelize the calculation
        
    Returns
    -------
    I : numpy.ndarray, 3d
        intensity map of the molecular transform
    """
    # generate asu mask and combine with resolution mask
    if mask is None:
        mask = np.ones(hkl_grid.shape[0]).astype(bool)
    mask *= get_asu_mask(model.space_group, hkl_grid)
    
    # sum over asus to compute scattering for unique reciprocal wedge
    q_grid = 2*np.pi*np.inner(model.A_inv.T, hkl_grid).T
    I_asu = np.zeros(q_grid.shape[0])
    for asu in range(model.xyz.shape[0]):
        I_asu[mask] += np.square(np.abs(structure_factors(q_grid[mask],
                                                          model.xyz[asu],
                                                          model.ff_a[asu], 
                                                          model.ff_b[asu], 
                                                          model.ff_c[asu], 
                                                          U=U, 
                                                          batch_size=batch_size,
                                                          n_processes=n_processes)))
        
    # get symmetry information for expanded map
    sym_ops = expand_sym_ops(model.sym_ops)
    hkl_sym = get_symmetry_equivalents(hkl_grid, sym_ops)
    ravel, map_shape_ravel = get_ravel_indices(hkl_sym, sampling)
    sampling_ravel = get_centered_sampling(map_shape_ravel, sampling)
    hkl_grid_mult, mult = compute_multiplicity(model, 
                                               sampling_ravel[0], 
                                               sampling_ravel[1], 
                                               sampling_ravel[2])

    # symmetrize and account for multiplicity
    I = np.zeros(map_shape_ravel).flatten()
    I[ravel[0]] = I_asu.copy()
    for asu in range(1, ravel.shape[0]):
        I[ravel[asu]] += I_asu.copy()
    I = I.reshape(map_shape_ravel)
    I /= (mult.max() / mult) 
    
    sampling_original = [(int(hkl_grid[:,i].min()),int(hkl_grid[:,i].max()),sampling[i]) for i in range(3)]
    I = resize_map(I, sampling_original, sampling_ravel)
    
    return I
    
@debug
def incoherent_sum_reciprocal(model, hkl_grid, sampling, U=None, batch_size=10000, n_processes=8):
    """
    Compute the incoherent sum of the scattering from all asus.
    For each grid point, the symmetry-equivalents are determined
    and mapped from 3d to 1d space by raveling. The intensities 
    for the first asu are computed and mapped to subsequent asus.
    Finally, intensities across symmetry-equivalent reflections 
    are summed (hence in reciprocal rather than real space). The 
    extents defined by hkl_grid are maintained.
    
    Parameters
    ----------
    model : AtomicModel
        instance of AtomicModel class expanded to p1
    hkl_grid : numpy.ndarray, shape (n_points, 3)
        hkl vectors of map grid points
    sampling : tuple
        sampling frequency along h,k,l axes
    U : numpy.ndarray, shape (n_atoms,)
        isotropic displacement parameters, applied to each asymmetric unit
    batch_size : int
        number of q-vectors to evaluate per batch
    n_processes : int
        number of processors over which to parallelize the calculation
        
    Returns
    -------
    I : numpy.ndarray, (n_points,)
        intensity map of the molecular transform
    """
    hkl_grid_sym = get_symmetry_equivalents(hkl_grid, model.sym_ops)
    ravel, map_shape_ravel = get_ravel_indices(hkl_grid_sym, sampling)
    
    I_sym = np.zeros(ravel.shape)
    for asu in range(I_sym.shape[0]):
        q_asu = 2*np.pi*np.inner(model.A_inv.T, hkl_grid_sym[asu]).T
        if asu == 0:
            I_sym[asu] = np.square(np.abs(structure_factors(q_asu,
                                                            model.xyz[0],
                                                            model.ff_a[0],
                                                            model.ff_b[0],
                                                            model.ff_c[0],
                                                            U=U,
                                                            batch_size=batch_size,
                                                            n_processes=n_processes)))
        else:
            intersect1d, comm1, comm2 = np.intersect1d(ravel[0], ravel[asu], return_indices=True)
            I_sym[asu][comm2] = I_sym[0][comm1]
            comm3 = np.arange(len(ravel[asu]))[~np.in1d(ravel[asu],ravel[0])]
            I_sym[asu][comm3] = np.square(np.abs(structure_factors(q_asu[comm3],
                                                                   model.xyz[0],
                                                                   model.ff_a[0],
                                                                   model.ff_b[0],
                                                                   model.ff_c[0],
                                                                   U=U,
                                                                   batch_size=batch_size)))
    I = np.sum(I_sym, axis=0)
    return I

"""
PyTorch implementation of transform calculations for diffuse scattering.

This module contains PyTorch versions of the transform calculation functions defined
in eryx/base.py. All implementations maintain the same API as the NumPy versions
but use PyTorch tensors and operations to enable gradient flow.

References:
    - Original NumPy implementation in eryx/base.py
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional, Union, Any

def compute_molecular_transform(pdb_path: str, 
                              hsampling: Tuple[float, float, float], 
                              ksampling: Tuple[float, float, float], 
                              lsampling: Tuple[float, float, float], 
                              U: Optional[torch.Tensor] = None, 
                              expand_p1: bool = True,
                              expand_friedel: bool = True, 
                              res_limit: float = 0, 
                              batch_size: int = 10000, 
                              n_processes: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the molecular transform as the incoherent sum of asymmetric units using PyTorch.
    
    Args:
        pdb_path: Path to coordinates file
        hsampling: Tuple (hmin, hmax, oversampling) for h dimension
        ksampling: Tuple (kmin, kmax, oversampling) for k dimension
        lsampling: Tuple (lmin, lmax, oversampling) for l dimension
        U: Optional tensor with isotropic displacement parameters
        expand_p1: If True, expand PDB to unit cell
        expand_friedel: If True, expand to full sphere in reciprocal space
        res_limit: High resolution limit in Angstrom
        batch_size: Number of q-vectors per batch
        n_processes: Number of processes (ignored in PyTorch implementation)
        
    Returns:
        Tuple containing:
            - PyTorch tensor of shape (n_points, 3) with q-vectors
            - PyTorch tensor of shape (dim_h, dim_k, dim_l) with intensity map
            
    References:
        - Original implementation: eryx/base.py:compute_molecular_transform
    """
    # TODO: Load model with adapter to convert to PyTorch tensors
    # TODO: Generate grid using map_utils_torch.generate_grid
    # TODO: Apply resolution mask
    # TODO: Choose calculation approach based on space group and expand_friedel
    # TODO: Call appropriate summation function
    
    raise NotImplementedError("compute_molecular_transform not implemented")

def compute_crystal_transform(pdb_path: str, 
                             hsampling: Tuple[float, float, float], 
                             ksampling: Tuple[float, float, float], 
                             lsampling: Tuple[float, float, float], 
                             U: Optional[torch.Tensor] = None, 
                             expand_p1: bool = True, 
                             res_limit: float = 0, 
                             batch_size: int = 5000, 
                             n_processes: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute crystal transform as coherent sum of asymmetric units using PyTorch.
    
    Args:
        pdb_path: Path to coordinates file
        hsampling: Tuple (hmin, hmax, oversampling) for h dimension
        ksampling: Tuple (kmin, kmax, oversampling) for k dimension
        lsampling: Tuple (lmin, lmax, oversampling) for l dimension
        U: Optional tensor with isotropic displacement parameters
        expand_p1: If True, expand PDB to unit cell
        res_limit: High resolution limit in Angstrom
        batch_size: Number of q-vectors per batch
        n_processes: Number of processes (ignored in PyTorch implementation)
        
    Returns:
        Tuple containing:
            - PyTorch tensor of shape (n_points, 3) with q-vectors
            - PyTorch tensor of shape (dim_h, dim_k, dim_l) with intensity map
            
    References:
        - Original implementation: eryx/base.py:compute_crystal_transform
    """
    # TODO: Load model with adapter
    # TODO: Generate grid and q-vectors
    # TODO: Apply resolution and dq masks
    # TODO: Compute structure factors for Bragg reflections
    # TODO: Reshape results to 3D map
    
    raise NotImplementedError("compute_crystal_transform not implemented")

def incoherent_sum_real(model: Any, 
                       hkl_grid: torch.Tensor, 
                       sampling: Tuple[float, float, float], 
                       U: Optional[torch.Tensor] = None, 
                       mask: Optional[torch.Tensor] = None, 
                       batch_size: int = 10000, 
                       n_processes: int = 8) -> torch.Tensor:
    """
    Compute incoherent sum of scattering from all ASUs in real space using PyTorch.
    
    Args:
        model: AtomicModel instance (converted to use PyTorch)
        hkl_grid: PyTorch tensor of shape (n_points, 3) with hkl indices
        sampling: Tuple with sampling rates
        U: Optional tensor with displacement parameters
        mask: Optional tensor with boolean mask
        batch_size: Number of q-vectors per batch
        n_processes: Number of processes (ignored in PyTorch implementation)
        
    Returns:
        PyTorch tensor of shape (dim_h, dim_k, dim_l) with intensity map
        
    References:
        - Original implementation: eryx/base.py:incoherent_sum_real
    """
    # TODO: Generate ASU mask and combine with resolution mask
    # TODO: Compute scattering for unique reciprocal wedge
    # TODO: Expand symmetry operations and hkl indices
    # TODO: Get raveled indices
    # TODO: Perform symmetrization using PyTorch operations
    # TODO: Account for multiplicity
    # TODO: Resize map if needed
    
    raise NotImplementedError("incoherent_sum_real not implemented")

def incoherent_sum_reciprocal(model: Any, 
                            hkl_grid: torch.Tensor, 
                            sampling: Tuple[float, float, float], 
                            U: Optional[torch.Tensor] = None, 
                            batch_size: int = 10000, 
                            n_processes: int = 8) -> torch.Tensor:
    """
    Compute incoherent sum of scattering in reciprocal space using PyTorch.
    
    Args:
        model: AtomicModel instance (converted to use PyTorch)
        hkl_grid: PyTorch tensor of shape (n_points, 3) with hkl indices
        sampling: Tuple with sampling rates
        U: Optional tensor with displacement parameters
        batch_size: Number of q-vectors per batch
        n_processes: Number of processes (ignored in PyTorch implementation)
        
    Returns:
        PyTorch tensor of shape (n_points,) with intensity values
        
    References:
        - Original implementation: eryx/base.py:incoherent_sum_reciprocal
    """
    # TODO: Get symmetry equivalents of hkl indices
    # TODO: Compute raveled indices
    # TODO: Initialize tensors for symmetry-equivalent structure factors
    # TODO: Compute structure factors for the first ASU
    # TODO: Map to other ASUs and compute additional structure factors
    # TODO: Sum intensities over all symmetry-equivalents
    
    raise NotImplementedError("incoherent_sum_reciprocal not implemented")
