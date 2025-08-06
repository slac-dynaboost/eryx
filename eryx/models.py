import numpy as np
import scipy.signal
import scipy.spatial
from scipy.linalg import block_diag
from tqdm import tqdm
from .pdb import AtomicModel, Crystal, GaussianNetworkModel
from .map_utils import *
from .scatter import structure_factors
from eryx.autotest.debug import debug

class OnePhonon:

    """
    Lattice of interacting rigid bodies in the one-phonon
    approximation (a.k.a small-coupling regime).
    """

    #@debug
    def __init__(self, pdb_path, hsampling, ksampling, lsampling,
                 expand_p1=True, group_by='asu',
                 res_limit=0., model='gnm',
                 gnm_cutoff=4., gamma_intra=1., gamma_inter=1.,
                 batch_size=10000, n_processes=8):
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self.batch_size = batch_size
        self.n_processes = n_processes
        self._setup(pdb_path, expand_p1, res_limit, group_by)
        self._setup_phonons(pdb_path, model,
                            gnm_cutoff, gamma_intra, gamma_inter)

    #@debug
    def _setup(self, pdb_path, expand_p1, res_limit, group_by):
        """
        Compute q-vectors to evaluate and build the unit cell
        and its nearest neighbors while storing useful dimensions.

        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        expand_p1: bool
            expand_p1 : bool
            if True, expand to p1 (i.e. if PDB corresponds to the asymmetric unit)
        res_limit : float
            high-resolution limit in Angstrom
        group_by : str
            level of rigid-body assembly.
            For now, only None and 'asu' have been implemented.
        """
        self.model = AtomicModel(pdb_path, expand_p1)

        self.hkl_grid, self.map_shape = generate_grid(self.model.A_inv,
                                                      self.hsampling,
                                                      self.ksampling,
                                                      self.lsampling,
                                                      return_hkl=True)
        self.res_mask, res_map = get_resolution_mask(self.model.cell,
                                                     self.hkl_grid,
                                                     res_limit)
        self.q_grid = 2 * np.pi * np.inner(self.model.A_inv.T, self.hkl_grid).T

        self.crystal = Crystal(self.model)
        self.crystal.supercell_extent(nx=1, ny=1, nz=1)
        self.id_cell_ref = self.crystal.hkl_to_id([0,0,0])
        self.n_cell = self.crystal.n_cell
        self.n_asu = self.crystal.model.n_asu
        self.n_atoms_per_asu = self.crystal.get_asu_xyz().shape[0]
        self.n_dof_per_asu_actual = self.n_atoms_per_asu * 3

        self.group_by = group_by
        if self.group_by is None:
            self.n_dof_per_asu = np.copy(self.n_dof_per_asu_actual)
        else:
            self.n_dof_per_asu = 6
        self.n_dof_per_cell = self.n_asu * self.n_dof_per_asu

    #@debug
    def _setup_phonons(self, pdb_path, model,
                       gnm_cutoff, gamma_intra, gamma_inter):
        """
        Compute phonons either from a Gaussian Network Model of the
        molecules or by direct definition of the dynamical matrix.

        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        model : str
            chosen phonon model: 'gnm' or 'rb'
        gnm_cutoff : float
            distance cutoff used to define the GNM
            see eryx.pdb.GaussianNetworkModel.compute_hessian()
        gamma_intra: float
            spring constant for atom pairs belonging to the same molecule
            see eryx.pdb.GaussianNetworkModel.build_gamma()
        gamma_inter: float
            spring constant for atom pairs belonging to distinct molecules
            see eryx.pdb.GaussianNetworkModel.build_gamma()
        """
        self.kvec = np.zeros((self.hsampling[2],
                              self.ksampling[2],
                              self.lsampling[2],
                              3))
        self.kvec_norm = np.zeros((self.hsampling[2],
                                   self.ksampling[2],
                                   self.lsampling[2],
                                   1))
        self.V = np.zeros((self.hsampling[2],
                           self.ksampling[2],
                           self.lsampling[2],
                           self.n_asu * self.n_dof_per_asu,
                           self.n_asu * self.n_dof_per_asu),
                          dtype='complex')
        self.Winv = np.zeros((self.hsampling[2],
                              self.ksampling[2],
                              self.lsampling[2],
                              self.n_asu * self.n_dof_per_asu),
                             dtype='complex')

        self._build_A()
        self._build_M()
        self._build_kvec_Brillouin()
        if model == 'gnm':
            self._setup_gnm(pdb_path, gnm_cutoff, gamma_intra, gamma_inter)
            self.compute_gnm_phonons()
            self.compute_covariance_matrix()
        else:
            self.compute_rb_phonons()

    def _setup_gnm(self, pdb_path, gnm_cutoff, gamma_intra, gamma_inter):
        """
        Instantiate the Gaussian Network Model

        Parameters
        ----------
        pdb_path : str
            path to coordinates file of asymmetric unit
        gnm_cutoff : float
            distance cutoff used to define the GNM
            see eryx.pdb.GaussianNetworkModel.compute_hessian()
        gamma_intra: float
            spring constant for atom pairs belonging to the same molecule
            see eryx.pdb.GaussianNetworkModel.build_gamma()
        gamma_inter: float
            spring constant for atom pairs belonging to distinct molecules
            see eryx.pdb.GaussianNetworkModel.build_gamma()
        """
        self.gnm = GaussianNetworkModel(pdb_path,
                                        enm_cutoff=gnm_cutoff,
                                        gamma_intra=gamma_intra,
                                        gamma_inter=gamma_inter)

    #@debug
    def _build_A(self):
        """
        Build the matrix A that projects small rigid-body displacements
        to the individual atoms in the rigid body.
        More specifically, consider the set of cartesian coordinates {r_i}_m
        of all atoms in the m-th rigid body and o_m their center of mass.
        Also consider their instantaneous displacement {u_i}_m translating
        from instantaneous rigid-body displacements w_m = [t_m, l_m] where
        t_m and l_m are respectively the 3-dimensional translation and libration
        vector of group m.
        For each atom i in group m, the conversion reads:
        u_i = A(r_i - o_m).w_m
        where A is the following 3x6 matrix:
        A(x,y,z) = [[ 1 0 0  0  z -y ]
                    [ 0 1 0 -z  0  x ]
                    [ 0 0 1  y -x  0 ]]
        """
        if self.group_by == 'asu':
            self.Amat = np.zeros((self.n_asu, self.n_atoms_per_asu, 3, 6))
            Atmp = np.zeros((3, 3))
            Adiag = np.copy(Atmp)
            np.fill_diagonal(Adiag, 1.)
            for i_asu in range(self.n_asu):
                xyz = np.copy(self.crystal.get_asu_xyz(i_asu))
                xyz -= np.mean(xyz, axis=0)
                
                # Debug print for centered coordinates
                if i_asu == 0:
                    print(f"NumPy ASU {i_asu} Centered XYZ (mean): {np.mean(xyz, axis=0)}")
                    print(f"NumPy ASU {i_asu} Centered XYZ (first 3 atoms):")
                    for i in range(min(3, xyz.shape[0])):
                        print(f"  Atom {i}: {xyz[i]}")
                
                for i_atom in range(self.n_atoms_per_asu):
                    # Debug print for specific atoms
                    if i_asu == 0 and i_atom < 3:
                        print(f"  NumPy Atom {i_atom} XYZ: {xyz[i_atom]}")
                    
                    # Create fresh Atmp for each atom
                    Atmp = np.zeros((3, 3))
                    
                    # Fill the skew-symmetric matrix
                    Atmp[0, 1] = xyz[i_atom, 2]
                    Atmp[0, 2] = -xyz[i_atom, 1]
                    Atmp[1, 2] = xyz[i_atom, 0]
                    
                    # Debug print before subtraction
                    if i_asu == 0 and i_atom < 3:
                        print(f"  NumPy Atom {i_atom} Atmp before subtraction:\n{Atmp}")
                    
                    # Apply skew-symmetry
                    Atmp -= Atmp.T
                    
                    # Debug print after subtraction
                    if i_asu == 0 and i_atom < 3:
                        print(f"  NumPy Atom {i_atom} Atmp after subtraction:\n{Atmp}")
                    
                    # Create the full block
                    stacked_block = np.hstack([Adiag, Atmp])
                    
                    # Debug print for stacked block
                    if i_asu == 0 and i_atom < 3:
                        print(f"  NumPy Atom {i_atom} Stacked Block:\n{stacked_block}")
                    
                    # Assign to Amat
                    self.Amat[i_asu, i_atom] = stacked_block
            
            # Reshape to final form
            self.Amat = self.Amat.reshape((self.n_asu,
                                           self.n_dof_per_asu_actual,
                                           self.n_dof_per_asu))
            
            # Debug prints for final Amat
            if self.n_asu > 0 and self.n_atoms_per_asu > 0:
                print(f"\nNumPy DEBUG _build_A: First ASU, first atom Amat block:")
                print(self.Amat[0, 0:3, :])
                if self.n_atoms_per_asu > 1:
                    print(f"\nNumPy DEBUG _build_A: First ASU, second atom Amat block:")
                    print(self.Amat[0, 3:6, :])
                if self.n_atoms_per_asu > 2:
                    print(f"\nNumPy DEBUG _build_A: First ASU, third atom Amat block:")
                    print(self.Amat[0, 6:9, :])
                
                # Print the entire Amat shape and a summary of its values
                print(f"\nNumPy DEBUG _build_A: Amat shape: {self.Amat.shape}")
                print(f"NumPy DEBUG _build_A: Amat min: {self.Amat.min()}, max: {self.Amat.max()}")
                print(f"NumPy DEBUG _build_A: Amat mean: {self.Amat.mean()}, std: {self.Amat.std()}")
        else:
            self.Amat = None

    #@debug
    def _build_M(self):
        """
        Build the mass matrix M.
        If all atoms are considered, M = M_0 is diagonal (see _build_M_allatoms())
        and Linv = 1./sqrt(M_0) is diagonal also.
        If atoms are grouped as rigid bodies, the all-atoms M matrix is
        projected using the A matrix: M = A.T M_0 A and Linv is obtained
        via Cholesky decomposition: M = LL.T
        """
        M_allatoms = self._build_M_allatoms()
        if self.group_by is None:
            M_allatoms = M_allatoms.reshape((self.n_asu * self.n_dof_per_asu_actual,
                                             self.n_asu * self.n_dof_per_asu_actual))
            self.Linv = 1. / np.sqrt(M_allatoms)
        else:
            Mmat = self._project_M(M_allatoms)
            Mmat = Mmat.reshape((self.n_asu * self.n_dof_per_asu,
                                 self.n_asu * self.n_dof_per_asu))
            self.Linv = np.linalg.inv(np.linalg.cholesky(Mmat))

    #@debug
    def _project_M(self, M_allatoms):
        """
        Project all-atom mass matrix M_0 using the A matrix: M = A.T M_0 A

        Parameters
        ----------
        M_allatoms : numpy.ndarray, shape (n_asu, n_atoms*3, n_asu, n_atoms*3)

        Returns
        -------
        Mmat: numpy.ndarray, shape (n_asu, n_dof_per_asu, n_asu, n_dof_per_asu)
        """
        Mmat = np.zeros((self.n_asu, self.n_dof_per_asu,
                         self.n_asu, self.n_dof_per_asu))
        for i_asu in range(self.n_asu):
            for j_asu in range(self.n_asu):
                Mmat[i_asu, :, j_asu, :] = \
                    np.matmul(self.Amat[i_asu].T,
                              np.matmul(M_allatoms[i_asu, :, j_asu, :],
                                        self.Amat[j_asu]))
        return Mmat

    #@debug
    def _build_M_allatoms(self):
        """
        Build all-atom mass matrix M_0

        Returns
        -------
        M_allatoms : numpy.ndarray, shape (n_asu, n_atoms*3, n_asu, n_atoms*3)
        """
        print("\n--- NumPy _build_M_allatoms Debug ---")
        
        # Extract weights
        mass_array = np.array([element.weight for structure in self.crystal.model.elements for element in structure])
        print(f"NumPy mass_array: shape={mass_array.shape}, dtype={mass_array.dtype}")
        print(f"NumPy mass_array stats: min={mass_array.min()}, max={mass_array.max()}, mean={mass_array.mean()}")
        print(f"NumPy mass_array first few elements: {mass_array[:5]}")
        
        # Create mass_list
        print("\nCreating NumPy mass_list...")
        mass_list = []
        for i in range(self.n_asu * self.n_atoms_per_asu):
            # Create 3x3 block for each atom (mass * identity)
            block = np.eye(3) * mass_array[i]
            mass_list.append(block)
        
        print(f"NumPy mass_list length: {len(mass_list)}")
        print(f"NumPy first block shape: {mass_list[0].shape}")
        print(f"NumPy first block:\n{mass_list[0]}")
        
        # Create block diagonal matrix
        print("\nCreating NumPy block diagonal matrix...")
        M_block_diag = block_diag(*mass_list)
        print(f"NumPy M_block_diag shape: {M_block_diag.shape}")
        print(f"NumPy M_block_diag diagonal elements (first few): {np.diagonal(M_block_diag)[:15]}")
        
        # Reshape to 4D tensor
        M_allatoms = M_block_diag.reshape((self.n_asu, self.n_dof_per_asu_actual,
                                           self.n_asu, self.n_dof_per_asu_actual))
        
        # Debug prints for M_allatoms
        print(f"\nNumPy M_allatoms shape: {M_allatoms.shape}")
        print(f"NumPy M_allatoms diag[0:5]: {np.diagonal(M_allatoms[0,:,0,:])[:5]}")
        if M_allatoms.shape[0] > 1 and M_allatoms.shape[2] > 1:
            print(f"NumPy M_allatoms[0,0,1,0]: {M_allatoms[0,0,1,0]}")
        
        return M_allatoms

    #@debug
    def _center_kvec(self, x, L):
        """
        For x and L integers such that 0 < x < L, return -L/2 < x < L/2
        by applying periodic boundary condition in L/2
        Parameters
        ----------
        x : int
            the index to center
        L : int
            length of the periodic box
        """
        return int(((x - L / 2) % L) - L / 2) / L

    #@debug
    def _build_kvec_Brillouin(self):
        """
        Compute all k-vectors and their norm in the first Brillouin zone.
        This is achieved by regularly sampling [-0.5,0.5[ for h, k and l.
        """
        for dh in range(self.hsampling[2]):
            k_dh = self._center_kvec(dh, self.hsampling[2])
            for dk in range(self.ksampling[2]):
                k_dk = self._center_kvec(dk, self.ksampling[2])
                for dl in range(self.lsampling[2]):
                    k_dl = self._center_kvec(dl, self.lsampling[2])
                    # Debug calculation for specific points
                    if dh == 0 and dk == 1 and dl == 0:
                        print(f"\nDEBUGGING NumPy calculation for point [0,1,0]:")
                        print(f"k_dh, k_dk, k_dl = {k_dh}, {k_dk}, {k_dl}")
                        print(f"hkl: {(k_dh, k_dk, k_dl)}")
                        print(f"A_inv:\n{self.model.A_inv}")
                        print(f"A_inv.T:\n{self.model.A_inv.T}")
                        result = np.inner(self.model.A_inv.T, (k_dh, k_dk, k_dl)).T
                        print(f"Result of np.inner: {result}")
                    
                    self.kvec[dh, dk, dl] = np.inner(self.model.A_inv.T,
                                                     (k_dh, k_dk, k_dl)).T
                    self.kvec_norm[dh, dk, dl] = np.linalg.norm(self.kvec[dh, dk, dl])

    #@debug
    def _at_kvec_from_miller_points(self, hkl_kvec):
        """
        Return the indices of all q-vector that are k-vector away from any
        Miller index in the map.

        Parameters
        ----------
        hkl_kvec : tuple of ints
            fractional Miller index of the desired k-vector
        """
        hsteps = int(self.hsampling[2] * (self.hsampling[1] - self.hsampling[0]) + 1)
        ksteps = int(self.ksampling[2] * (self.ksampling[1] - self.ksampling[0]) + 1)
        lsteps = int(self.lsampling[2] * (self.lsampling[1] - self.lsampling[0]) + 1)

        index_grid = np.mgrid[
                     hkl_kvec[0]:hsteps:self.hsampling[2],
                     hkl_kvec[1]:ksteps:self.ksampling[2],
                     hkl_kvec[2]:lsteps:self.lsampling[2]]

        return np.ravel_multi_index((index_grid[0].flatten(),
                                     index_grid[1].flatten(),
                                     index_grid[2].flatten()),
                                    self.map_shape)

    def _kvec_map(self):
        """
        Build a map where the intensity at each fractional Miller index
        is set as the norm of the corresponding k-vector.
        For example, at each integral Miller index, k-vector is zero and
        it increases and then decreases as we sample between them.

        Returns
        -------
        map : numpy.ndarray, shape (npoints, 1)
        """
        map = np.zeros((self.q_grid.shape[0]))
        for dh in tqdm(range(self.hsampling[2])):
            for dk in range(self.ksampling[2]):
                for dl in range(self.lsampling[2]):
                    q_indices = self._at_kvec_from_miller_points((dh, dk, dl))
                    map[q_indices] = np.linalg.norm(self.kvec[dh, dk, dl])
        return map

    def _q_map(self):
        """
        Build a map where the intensity at each fractional Miller index
        is set as the norm of the corresponding q-vector.

        Returns
        -------
        map : numpy.ndarray, shape (npoints, 1_
        """
        map = np.zeros((self.q_grid.shape[0]))
        for dh in tqdm(range(self.hsampling[2])):
            for dk in range(self.ksampling[2]):
                for dl in range(self.lsampling[2]):
                    q_indices = self._at_kvec_from_miller_points((dh, dk, dl))
                    map[q_indices] = np.linalg.norm(self.q_grid[q_indices], axis=1)
        return map

    #@debug
    def compute_hessian(self):
        """
        Build the projected Hessian matrix for the supercell.

        Returns
        -------
        hessian : numpy.ndarray,
                  shape (n_asu, n_dof_per_asu, n_cell, n_asu, n_dof_per_asu),
                  dtype 'complex'
            Hessian matrix for the assembly of rigid bodies in the supercell.
        """
        hessian = np.zeros((self.n_asu, self.n_dof_per_asu,
                            self.n_cell, self.n_asu, self.n_dof_per_asu),
                           dtype='complex')

        hessian_allatoms = self.gnm.compute_hessian()

        for i_cell in range(self.n_cell):
            for i_asu in range(self.n_asu):
                for j_asu in range(self.n_asu):
                    hessian[i_asu, :, i_cell, j_asu, :] = \
                        np.matmul(self.Amat[i_asu].T,
                                  np.matmul(np.kron(hessian_allatoms[i_asu, :, i_cell, j_asu, :],
                                                    np.eye(3)),
                                            self.Amat[j_asu]))

        return hessian

    #@debug
    def compute_covariance_matrix(self):
        """
        Compute covariance matrix for all asymmetric units.
        The covariance matrix results from modelling pairwise
        interactions with a Gaussian Network Model where atom
        pairs belonging to different asymmetric units are not
        interacting. It is scaled to match the ADPs in the input PDB file.
        """
        self.covar = np.zeros((self.n_asu*self.n_dof_per_asu,
                               self.n_cell, self.n_asu*self.n_dof_per_asu),
                              dtype='complex')

        hessian = self.compute_hessian()
        for dh in range(self.hsampling[2]):
            for dk in range(self.ksampling[2]):
                for dl in range(self.lsampling[2]):
                    kvec = self.kvec[dh,dk,dl]
                    Kinv = self.gnm.compute_Kinv(hessian, kvec=kvec, reshape=False)
                    for j_cell in range(self.n_cell):
                        r_cell = self.crystal.get_unitcell_origin(self.crystal.id_to_hkl(j_cell))
                        phase = np.dot(kvec, r_cell)
                        eikr = np.cos(phase) + 1j * np.sin(phase)
                        self.covar[:,j_cell,:] += Kinv * eikr
        #ADP_scale = np.mean(self.model.adp[0]) / \
        #            (8 * np.pi * np.pi * np.mean(np.diag(self.covar[:,self.crystal.hkl_to_id([0,0,0]),:])) / 3.)
        #self.covar *= ADP_scale
        self.ADP = np.real(np.diag(self.covar[:,self.crystal.hkl_to_id([0,0,0]),:]))
        Amat = np.transpose(self.Amat, (1,0,2)).reshape(self.n_dof_per_asu_actual, self.n_asu*self.n_dof_per_asu)
        self.ADP = Amat @ self.ADP
        self.ADP = np.sum(self.ADP.reshape(int(self.ADP.shape[0]/3),3),axis=1)
        ADP_scale = np.mean(self.model.adp) / (8*np.pi*np.pi*np.mean(self.ADP)/3)
        self.ADP *= ADP_scale
        self.covar *= ADP_scale
        self.covar = np.real(self.covar.reshape((self.n_asu, self.n_dof_per_asu,
                                                 self.n_cell, self.n_asu, self.n_dof_per_asu)))

    #@debug
    def compute_gnm_phonons(self):
        """
        Compute the dynamical matrix for each k-vector in the first
        Brillouin zone, from the supercell's GNM.
        The squared inverse of their eigenvalues is
        stored for intensity calculation and their eigenvectors are
        mass-weighted to be used in the definition of the phonon
        structure factors.
        """
        hessian = self.compute_hessian()
        for dh in range(self.hsampling[2]):
            for dk in range(self.ksampling[2]):
                for dl in range(self.lsampling[2]):
                    Kmat = self.gnm.compute_K(hessian, kvec=self.kvec[dh, dk, dl])
                    Kmat = Kmat.reshape((self.n_asu * self.n_dof_per_asu,
                                         self.n_asu * self.n_dof_per_asu))
                    Dmat = np.matmul(self.Linv,
                                     np.matmul(Kmat, self.Linv.T))
                    v, w, _ = np.linalg.svd(Dmat)
                    w = np.sqrt(w)
                    w = np.where(w < 1e-6, np.nan, w)
                    w = w[::-1]
                    v = v[:,::-1]
                    self.Winv[dh, dk, dl] = 1. / w ** 2
                    self.V[dh, dk, dl] = np.matmul(self.Linv.T, v)

    def compute_rb_phonons(self):
        """
        Compute the dynamical matrix for each k-vector in the first
        Brillouin zone as a decaying Gaussian of k.
        (in development, not fully tested or understood).
        """
        Kmat = np.zeros((self.n_asu * self.n_dof_per_asu,
                         self.n_asu * self.n_dof_per_asu))
        for dh in range(self.hsampling[2]):
            for dk in range(self.ksampling[2]):
                for dl in range(self.lsampling[2]):
                    np.fill_diagonal(Kmat,
                                     np.exp(-0.5 * (
                                             np.linalg.norm(self.kvec[dh, dk, dl] /
                                                            np.linalg.norm(self.kvec[3, 0, 0])) ** 2))
                                     )
                    u, s, _ = np.linalg.svd(Kmat)
                    self.Winv[dh, dk, dl] = s
                    self.V[dh, dk, dl] = u

    #@debug
    def apply_disorder(self, rank=-1, outdir=None, use_data_adp=False):
        """
        Compute the diffuse intensity in the one-phonon scattering
        disorder model originating from a Gaussian Network Model
        representation of the asymmetric units, optionally reduced
        to a set of interacting rigid bodies.
        """
        import numpy as np  # Import numpy locally to avoid reference error
        if use_data_adp:
            ADP = self.model.adp[0] / (8 * np.pi * np.pi)
        else:
            ADP = self.ADP
        Id = np.zeros((self.q_grid.shape[0]), dtype='complex')
        for dh in tqdm(range(self.hsampling[2])):
            for dk in range(self.ksampling[2]):
                for dl in range(self.lsampling[2]):

                    # Debug print for specific BZ point
                    if dh == 0 and dk == 1 and dl == 0:
                        print(f"\n--- NumPy Debug for BZ point (0,1,0) ---")
                        print(f"V[0,1,0] shape: {self.V[0,1,0].shape}")
                        print(f"Winv[0,1,0] shape: {self.Winv[0,1,0].shape}")
                        print(f"V[0,1,0][0,0] (abs): {np.abs(self.V[0,1,0][0,0]):.8e}")
                        print(f"Winv[0,1,0][0]: {self.Winv[0,1,0][0]:.8e}")
                    
                    q_indices = self._at_kvec_from_miller_points((dh, dk, dl))
                    
                    # Debug print for specific BZ point
                    if dh == 0 and dk == 1 and dl == 0:
                        print(f"q_indices shape: {q_indices.shape}")
                        if q_indices.size > 0:
                            print(f"First few q_indices: {q_indices[:5]}")
                    
                    q_indices = q_indices[self.res_mask[q_indices]]
                    
                    # Debug print for specific BZ point
                    if dh == 0 and dk == 1 and dl == 0:
                        print(f"valid q_indices shape: {q_indices.shape}")
                        if q_indices.size > 0:
                            print(f"First few valid q_indices: {q_indices[:5]}")

                    F = np.zeros((q_indices.shape[0],
                                  self.n_asu,
                                  self.n_dof_per_asu),
                                 dtype='complex')
                    for i_asu in range(self.n_asu):
                        F[:, i_asu, :] = structure_factors(
                            self.q_grid[q_indices],
                            self.model.xyz[i_asu],
                            self.model.ff_a[i_asu],
                            self.model.ff_b[i_asu],
                            self.model.ff_c[i_asu],
                            U=ADP,
                            batch_size=self.batch_size,
                            n_processes=self.n_processes,
                            compute_qF=True,
                            project_on_components=self.Amat[i_asu],
                            sum_over_atoms=False)
                    F = F.reshape((q_indices.shape[0],
                                   self.n_asu * self.n_dof_per_asu))
                    
                    # Debug print for specific BZ point
                    if dh == 0 and dk == 1 and dl == 0 and q_indices.size > 0:
                        print(f"F shape: {F.shape}")
                        print(f"F[0,0] (abs): {np.abs(F[0,0]):.8e}" if F.size > 0 else "F is empty")

                    if rank == -1:
                        # Debug print for specific BZ point
                        if dh == 0 and dk == 1 and dl == 0 and q_indices.size > 0:
                            FV = np.dot(F, self.V[dh, dk, dl])
                            FV_abs_squared = np.square(np.abs(FV))
                            print(f"FV shape: {FV.shape}")
                            print(f"FV_abs_squared shape: {FV_abs_squared.shape}")
                            print(f"Winv shape: {self.Winv[dh, dk, dl].shape}")
                            print(f"FV[0,0] (abs): {np.abs(FV[0,0]):.8e}" if FV.size > 0 else "FV is empty")
                            print(f"FV_abs_squared[0,0]: {FV_abs_squared[0,0]:.8e}" if FV_abs_squared.size > 0 else "FV_abs_squared is empty")
                            print(f"Winv[0]: {self.Winv[dh, dk, dl][0]:.8e}")
                            intensity = np.dot(FV_abs_squared, self.Winv[dh, dk, dl])
                            print(f"intensity shape: {intensity.shape}")
                            print(f"intensity[0]: {intensity[0]:.8e}" if intensity.size > 0 else "intensity is empty")
                            Id[q_indices] += intensity
                        else:
                            Id[q_indices] += np.dot(
                                np.square(np.abs(np.dot(F, self.V[dh, dk, dl]))),
                                self.Winv[dh, dk, dl])
                    else:
                        Id[q_indices] += np.square(
                            np.abs(np.dot(F, self.V[dh,dk,dl,:,rank]))) * \
                                         self.Winv[dh,dk,dl,rank]
        Id[~self.res_mask] = np.nan
        Id = np.real(Id)
        if outdir is not None:
            import os
            import numpy as np
            np.save(os.path.join(outdir, f"rank_{rank:05}.npy"), Id)
        return Id

