import numpy as np
import gemmi
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eryx.pdb import *

class TestPDB(object):
    """
    Check correctness of PDB extraction methods.
    """
    @classmethod
    def setup_class(cls):
        cls.am_p212121 = AtomicModel("histidine.pdb", expand_p1=True)
        cls.am_p1 = AtomicModel("histidine_p1.pdb")
        cls.n_asus = 4
        
    def test_get_xyz_asus(self):
        """ Check that asu coordinates are correctly calculated. """
        asu = {i:self.am_p1.xyz[i*21:i*21+21,:] for i in range(self.n_asus)}
        match = list()
        for i in range(self.am_p212121.xyz.shape[0]):
            arr = np.array([np.sum(self.am_p212121.xyz[i] - asu[j]) for j in asu.keys()])
            match.append(np.where(np.around(arr, 6)==0)[0][0])
        match = np.sort(np.array(match))
        assert np.allclose(match, np.arange(self.n_asus))
