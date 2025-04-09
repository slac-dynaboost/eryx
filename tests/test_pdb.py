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
    def setup_class(cls):
        cls.pdb_ids = ["5zck", "7n2h", "193l"] # P 21 21 21, P 31, P 43 21 2
