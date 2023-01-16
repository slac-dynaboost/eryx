import numpy as np
import scipy.signal
import scipy.spatial
from .pdb import AtomicModel
from .map_utils import *
from .scatter import structure_factors
from .stats import compute_cc

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

class TranslationalDisorder:
    
    def __init__(self, pdb_path, hsampling, ksampling, lsampling, expand_friedel=True, res_limit=0, batch_size=10000, n_processes=8):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self._setup(pdb_path, expand_friedel, res_limit, batch_size, n_processes)
        
    def _setup(self, pdb_path, expand_friedel=True, res_limit=0, batch_size=10000, n_processes=8):
        """
        Set up class, including computing the molecular transform.
        
        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        expand_friedel : bool
            if True, expand to include portion of reciprocal space related by Friedel's law
        res_limit : float
            high resolution limit
        batch_size : int     
            number of q-vectors to evaluate per batch
        n_processes : int
            number of processors for structure factor calculation
        """
        self.q_grid, self.transform = compute_molecular_transform(pdb_path, 
                                                                  self.hsampling, 
                                                                  self.ksampling, 
                                                                  self.lsampling,
                                                                  expand_friedel=expand_friedel,
                                                                  res_limit=res_limit,
                                                                  batch_size=batch_size,
                                                                  n_processes=n_processes)
        self.transform[self.transform==0] = np.nan # compute_cc expects masked values to be np.nan
        self.q_mags = np.linalg.norm(self.q_grid, axis=1)
        self.map_shape = self.transform.shape
    
    def apply_disorder(self, sigmas):
        """
        Compute the diffuse map(s) from the molecular transform:
        I_diffuse = I_transform * (1 - q^2 * sigma^2)
        for a single sigma or set of (an)isotropic sigmas.

        Parameters
        ----------
        sigma : float or array of shape (n_sigma,) or (n_sigma, 3)
            (an)isotropic displacement parameter for asymmetric unit 

        Returns
        -------
        Id : numpy.ndarray, (n_sigma, q_grid.shape[0])
            diffuse intensity maps for the corresponding sigma(s)
        """
        if type(sigmas) == float:
            sigmas = np.array([sigmas])

        if len(sigmas.shape) == 1:
            wilson = np.square(self.q_mags) * np.square(sigmas)[:,np.newaxis]
        else:
            wilson = np.sum(self.q_grid.T * np.dot(np.square(sigmas)[:,np.newaxis] * np.eye(3), self.q_grid.T), axis=1)

        Id = self.transform.flatten() * (1 - np.exp(-1 * wilson))
        return Id
    
    def optimize(self, target, sigmas_min, sigmas_max, n_search=20):
        """
        Scan to find the sigma that maximizes the overall Pearson
        correlation between the target and computed maps. 
        
        Parameters
        ----------
        target : numpy.ndarray, 3d
            target map, of shape self.map_shape
        sigmas_min : float or tuple of shape (3,)
            lower bound of (an)isotropic sigmas
        sigmas_max : float or tuple of shape (3,)
            upper bound of (an)isotropic sigmas, same type/dimension as sigmas_min
        n_search : int
            sampling frequency between sigmas_min and sigmas_max
        
        Returns
        -------
        ccs : numpy.ndarray, shape (n_search,)
            Pearson correlation coefficients to target maps
        sigmas : numpy.ndarray, shape (n_search,) or (n_search, n_search, n_search)
            sigmas that were scanned over, ordered as ccs
        """
        assert target.shape == self.map_shape
        
        if (type(sigmas_min) == float) and (type(sigmas_max) == float):
            sigmas = np.linspace(sigmas_min, sigmas_max, n_search)
        else:
            sa, sb, sc = [np.linspace(sigmas_min[i], sigmas_max[i], n_search) for i in range(3)]
            sigmas = np.array(list(itertools.product(sa, sb, sc)))
        
        Id = self.apply_disorder(sigmas)
        ccs = compute_cc(Id, np.expand_dims(target.flatten(), axis=0))
        opt_index = np.argmax(ccs)
        self.opt_sigma = sigmas[opt_index]
        self.opt_map = Id[opt_index].reshape(self.map_shape)

        print(f"Optimal sigma: {self.opt_sigma}, with correlation coefficient {ccs[opt_index]:.4f}")
        return ccs, sigmas

class LiquidLikeMotions:
    
    """
    Model in which collective motions decay exponentially with distance
    across the crystal. Mathematically the predicted diffuse scattering 
    is the convolution between the crystal transform and a disorder kernel.
    """
    
    def __init__(self, pdb_path, hsampling, ksampling, lsampling, expand_p1=True, 
                 border=1, res_limit=0, batch_size=5000, n_processes=8):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self._setup(pdb_path, expand_p1, border, res_limit, batch_size, n_processes)
                
    def _setup(self, pdb_path, expand_p1, border, res_limit, batch_size, n_processes):
        """
        Set up class, including calculation of the crystal transform.
        The transform can be evaluated to a higher resolution so that
        the edge of the disorder map doesn't encounter a boundary.
        
        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        expand_p1 : bool
            if True, pdb corresponds to asymmetric unit; expand to unit cell
        border : int
            number of border (integral) Miller indices along each direction 
        res_limit : float
            high-resolution limit in Angstrom
        batch_size : int
            number of q-vectors to evaluate per batch
        n_processes : int
            number of processes for structure factor calculation
        """
        # generate atomic model
        model = AtomicModel(pdb_path, expand_p1=expand_p1)
        model.flatten_model()
        
        # get grid for padded map
        hsampling_padded = (self.hsampling[0]-border, self.hsampling[1]+border, self.hsampling[2])
        ksampling_padded = (self.ksampling[0]-border, self.ksampling[1]+border, self.ksampling[2])
        lsampling_padded = (self.lsampling[0]-border, self.lsampling[1]+border, self.lsampling[2])
        hkl_grid, self.map_shape = generate_grid(model.A_inv, 
                                                 hsampling_padded,
                                                 ksampling_padded,
                                                 lsampling_padded,
                                                 return_hkl=True)
        self.res_mask, res_map = get_resolution_mask(model.cell, hkl_grid, res_limit)
        
        # compute crystal transform
        self.q_grid, self.transform = compute_crystal_transform(pdb_path,
                                                                hsampling_padded,
                                                                ksampling_padded,
                                                                lsampling_padded,
                                                                expand_p1=expand_p1,
                                                                res_limit=res_limit,
                                                                batch_size=batch_size,
                                                                n_processes=n_processes)
        self.q_mags = np.linalg.norm(self.q_grid, axis=1)
        
        # generate mask for padded region
        self.mask = np.zeros(self.map_shape)
        self.mask[border*self.hsampling[2]:-border*self.hsampling[2],
                  border*self.ksampling[2]:-border*self.ksampling[2],
                  border*self.lsampling[2]:-border*self.lsampling[2]] = 1
        self.map_shape_nopad = tuple(np.array(self.map_shape) - np.array([2*border*self.hsampling[2], 
                                                                          2*border*self.ksampling[2], 
                                                                          2*border*self.lsampling[2]]))
        
    def apply_disorder(self, sigmas, gammas):
        """
        Compute the diffuse map(s) from the crystal transform as:
        I_diffuse = q2s2 * np.exp(-q2s2) * [I_transform * kernel(q)]
        where q2s2 = q^2 * s^2, and the kernel models covariances as 
        decaying exponentially with interatomic distance: 
        kernel(q) = 8 * pi * gamma^3 / (1 + q^2 * gamma^2)^2
        
        Parameters
        ----------
        sigmas : float or array of shape (n_sigma,) or (n_sigma, 3)
            (an)isotropic displacement parameter for asymmetric unit 
        gammas : float or array of shape (n_gamma,)
            kernel's correlation length
            
        Returns
        -------
        Id : numpy.ndarray, (n_sigma*n_gamma, q_grid.shape[0])
            diffuse intensity maps for the corresponding parameters
        """
        
        if type(gammas) == float or type(gammas) == int:
            gammas = np.array([gammas])   
        if type(sigmas) == float or type(sigmas) == int:
            sigmas = np.array([sigmas])

        # generate kernel and convolve with transform
        Id = np.zeros((len(gammas), self.q_grid.shape[0]))
        kernels = 8.0 * np.pi * (gammas[:,np.newaxis]**3) / np.square(1 + np.square(gammas[:,np.newaxis] * self.q_mags))
        for num in range(len(gammas)):
            Id[num] = scipy.signal.fftconvolve(self.transform, kernels[num].reshape(self.map_shape), mode='same').flatten()
        Id = np.tile(Id, (len(sigmas), 1))

        # scale with displacement parameters
        if len(sigmas.shape)==1:
            sigmas = np.repeat(sigmas, len(gammas))
            q2s2 = np.square(self.q_mags) * np.square(sigmas)[:,np.newaxis]
        else:
            sigmas = np.repeat(sigmas, len(gammas), axis=0)
            q2s2 = np.sum(self.q_grid.T * np.dot(np.square(sigmas)[:,np.newaxis] * np.eye(3), self.q_grid.T), axis=1)

        Id *= np.exp(-1*q2s2) * q2s2
        Id[:,~self.res_mask] = np.nan
        return Id

    def optimize(self, target, sigmas_min, sigmas_max, gammas_min, gammas_max, ns_search=20, ng_search=10):
        """
        Scan to find the sigma that maximizes the overall Pearson
        correlation between the target and computed maps. 
        
        Parameters
        ----------
        target : numpy.ndarray, 3d
            target map, of shape self.map_shape
        sigmas_min : float or tuple of shape (3,)
            lower bound of (an)isotropic sigmas
        sigmas_max : float or tuple of shape (3,)
            upper bound of (an)isotropic sigmas, same type/dimension as sigmas_min
        gammas_min : float 
            lower bound of gamma
        gammas_max : float 
            upper bound of gamma
        ns_search : int
            sampling frequency between sigmas_min and sigmas_max
        ng_search : int
            sampling frequency between gammas_min and gammas_max
        
        Returns
        -------
        ccs : numpy.ndarray, shape (n_search,)
            Pearson correlation coefficients to target maps
        sigmas : numpy.ndarray, shape (n_search,) or (n_search, n_search, n_search)
            sigmas that were scanned over, ordered as ccs
        """
        assert target.shape == self.map_shape_nopad
        
        if (type(sigmas_min) == float) and (type(sigmas_max) == float):
            sigmas = np.linspace(sigmas_min, sigmas_max, ns_search)
        else:
            sa, sb, sc = [np.linspace(sigmas_min[i], sigmas_max[i], ns_search) for i in range(3)]
            sigmas = np.array(list(itertools.product(sa, sb, sc)))
        gammas = np.linspace(gammas_min, gammas_max, ng_search)
        
        Id = self.apply_disorder(sigmas, gammas)
        Id = Id[:,self.mask.flatten()==1]
        ccs = compute_cc(Id, np.expand_dims(target.flatten(), axis=0))
        opt_index = np.argmax(ccs)
        self.opt_map = Id[opt_index].reshape(self.map_shape_nopad)
        
        sigmas = np.repeat(sigmas, len(gammas), axis=0)
        gammas = np.tile(gammas, len(sigmas))
        self.opt_sigma = sigmas[opt_index]
        self.opt_gamma = gammas[opt_index]

        print(f"Optimal sigma: {self.opt_sigma}, optimal gamma: {self.opt_gamma}, with correlation coefficient {ccs[opt_index]:.4f}")
        return ccs, sigmas, gammas
    
class RotationalDisorder:
    
    """
    Model of rigid body rotational disorder, in which all atoms in 
    each asymmetric unit rotate as a rigid unit around a randomly 
    oriented axis with a normally distributed rotation angle.
    """
    
    def __init__(self, pdb_path, hsampling, ksampling, lsampling, expand_p1=True, res_limit=0, batch_size=10000, n_processes=8):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self._setup(pdb_path, expand_p1, res_limit)
        self.batch_size = batch_size
        self.n_processes = n_processes 
        
    def _setup(self, pdb_path, expand_p1, res_limit=0):
        """
        Compute q-vectors to evaluate.
        
        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        expand_p1 : bool
            if True, expand to p1 (i.e. if PDB corresponds to the asymmetric unit)
        res_limit : float
            high-resolution limit in Angstrom
        """
        self.model = AtomicModel(pdb_path, expand_p1=expand_p1, frame=-1)
        hkl_grid, self.map_shape = generate_grid(self.model.A_inv, 
                                                 self.hsampling, 
                                                 self.ksampling, 
                                                 self.lsampling,
                                                 return_hkl=True)
        self.q_grid = 2*np.pi*np.inner(self.model.A_inv.T, hkl_grid).T
        self.q_mags = np.linalg.norm(self.q_grid, axis=1)
        self.mask, res_map = get_resolution_mask(self.model.cell, hkl_grid, res_limit)
    
    @staticmethod
    def generate_rotations_around_axis(sigma, num_rot, axis=np.array([0,0,1.0])):
        """
        Generate uniform random rotations about an axis.
        Parameters
        ----------
        sigma : float
            standard deviation of angular sampling around axis in degrees
        num_rot : int
            number of rotations to generate
        axis : numpy.ndarray, shape (3,)
            axis about which to generate rotations
        Returns
        -------
        rot_mat : numpy.ndarray, shape (num, 3, 3)
            rotation matrices
        """
        axis /= np.linalg.norm(axis)
        random_R = scipy.spatial.transform.Rotation.random(num_rot).as_matrix()
        random_ax = np.inner(random_R, np.array([0,0,1.0]))
        thetas = np.deg2rad(sigma) * np.random.randn(num_rot)
        rot_vec = thetas[:,np.newaxis] * random_ax
        rot_mat = scipy.spatial.transform.Rotation.from_rotvec(rot_vec).as_matrix()
        return rot_mat
    
    def apply_disorder(self, sigmas, num_rot=100):
        """
        Compute the diffuse maps(s) resulting from rotational disorder for 
        the given sigmas by applying Guinier's equation to an ensemble of 
        rotated molecules, and then taking the incoherent sum of all of the
        asymmetric units.
        
        Parameters
        ----------
        sigmas : float or array of shape (n_sigma,) 
            standard deviation(s) of angular sampling in degrees
        num_rot : int
            number of rotations to generate per sigma
        
        Returns
        -------
        Id : numpy.ndarray, (n_sigma, q_grid.shape[0])
            diffuse intensity maps for the corresponding parameters
        """
        if type(sigmas) == float or type(sigmas) == int:
            sigmas = np.array([sigmas])
            
        Id = np.zeros((len(sigmas), self.q_grid.shape[0]))
        for n_sigma,sigma in enumerate(sigmas):
            for asu in range(self.model.n_asu):
                # rotate atomic coordinates
                rot_mat = self.generate_rotations_around_axis(sigma, num_rot)
                com = np.mean(self.model.xyz[asu], axis=0)
                xyz_rot = np.matmul(self.model.xyz[asu] - com, rot_mat) 
                xyz_rot += com
                
                # apply Guinier's equation to rotated ensemble
                fc = np.zeros(self.q_grid.shape[0], dtype=complex)
                fc_square = np.zeros(self.q_grid.shape[0])
                for rnum in range(num_rot):
                    A = structure_factors(self.q_grid[self.mask], 
                                          xyz_rot[rnum], 
                                          self.model.ff_a[asu], 
                                          self.model.ff_b[asu], 
                                          self.model.ff_c[asu], 
                                          U=None, 
                                          batch_size=self.batch_size,
                                          n_processes=self.n_processes)
                    fc[self.mask] += A
                    fc_square[self.mask] += np.square(np.abs(A)) 
                Id[n_sigma] += fc_square / num_rot - np.square(np.abs(fc / num_rot))

        Id[:,~self.mask] = np.nan
        return Id 
    
    def optimize(self, target, sigma_min, sigma_max, n_search=20, num_rot=100):
        """
        Scan to find the sigma that maximizes the overall Pearson
        correlation between the target and computed maps. 
        
        Parameters
        ----------
        target : numpy.ndarray, 3d
            target map, of shape self.map_shape
        sigma_min : float 
            lower bound of sigma
        sigma_max : float 
            upper bound of sigma
        n_search : int
            sampling frequency between sigma_min and sigma_max
        num_rot : int
            number of rotations to generate per sigma
        
        Returns
        -------
        ccs : numpy.ndarray, shape (n_search,)
            Pearson correlation coefficients to target maps
        sigmas : numpy.ndarray, shape (n_search,) or (n_search, n_search, n_search)
            sigmas that were scanned over, ordered as ccs
        """
        assert target.shape == self.map_shape
        
        sigmas = np.linspace(sigma_min, sigma_max, n_search)        
        Id = self.apply_disorder(sigmas, num_rot)
        ccs = compute_cc(Id, np.expand_dims(target.flatten(), axis=0))
        opt_index = np.argmax(ccs)
        self.opt_sigma = sigmas[opt_index]
        self.opt_map = Id[opt_index].reshape(self.map_shape)

        print(f"Optimal sigma: {self.opt_sigma}, with correlation coefficient {ccs[opt_index]:.4f}")
        return ccs, sigmas
    
class EnsembleDisorder:
    
    """
    Model of ensemble disorder, in which the components of the
    asymmetric unit populate distinct biological states.
    """
    
    def __init__(self, pdb_path, hsampling, ksampling, lsampling, batch_size=10000, expand_p1=True):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self._setup(pdb_path, expand_p1)
        self.batch_size = batch_size
        
    def _setup(self, pdb_path, expand_p1):
        """
        Compute q-vectors to evaluate.
        
        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        expand_p1 : bool
            if True, expand to p1 (i.e. if PDB corresponds to the asymmetric unit)
        """
        self.model = AtomicModel(pdb_path, expand_p1=expand_p1, frame=-1)
        self.q_grid, self.map_shape = generate_grid(self.model.A_inv, 
                                                    self.hsampling, 
                                                    self.ksampling, 
                                                    self.lsampling)
        self.q_mags = np.linalg.norm(self.q_grid, axis=1)
        
    def apply_disorder(self, weights=None):
        """
        Compute the diffuse maps(s) resulting from ensemble disorder using
        Guinier's equation, and then taking the incoherent sum of all the
        asymmetric units. 
        
        Parameters
        ----------
        weights : shape (n_sets, n_conf) 
            set(s) of probabilities associated with each conformation

        Returns
        -------
        Id : numpy.ndarray, (n_sets, q_grid.shape[0])
            diffuse intensity map for the corresponding parameters
        """
        if weights is None:
            weights = 1.0 / self.model.n_conf * np.array([np.ones(self.model.n_conf)])
        if len(weights.shape) == 1:
            weights = np.array([weights])
        if weights.shape[1] != self.model.n_conf:
            raise ValueError("Second dimension of weights must match number of conformations.")
            
        n_maps = weights.shape[0]
        Id = np.zeros((weights.shape[0], self.q_grid.shape[0]))

        for asu in range(self.model.n_asu):

            fc = np.zeros((weights.shape[0], self.q_grid.shape[0]), dtype=complex)
            fc_square = np.zeros((weights.shape[0], self.q_grid.shape[0]))

            for conf in range(self.model.n_conf):
                index = conf * self.model.n_asu + asu
                A = structure_factors(self.q_grid, 
                                      self.model.xyz[index], 
                                      self.model.ff_a[index], 
                                      self.model.ff_b[index], 
                                      self.model.ff_c[index], 
                                      U=None, 
                                      batch_size=10000)
                for nm in range(n_maps):
                    fc[nm] += A * weights[nm][conf]
                    fc_square[nm] += np.square(np.abs(A)) * weights[nm][conf]

            for nm in range(n_maps):
                Id[nm] += fc_square[nm] - np.square(np.abs(fc[nm]))
                
        return Id
