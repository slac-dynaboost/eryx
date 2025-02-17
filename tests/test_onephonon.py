import pytest
import numpy as np
from tests.test_utils.log_analysis import LogAnalyzer

class TestOnePhonon:

    @pytest.fixture
    def base_log_file(self):
        return "tests/test_data/logs/base_run/base_run.log"

    @pytest.fixture
    def edge_log_file(self):
        return "tests/test_data/logs/edge_cases/edge_run_2.log"

    def test_input_parameters(self, base_log_file):
        """Test that input parameters are correctly parsed from log"""
        with open(base_log_file) as f:
            log_content = f.read()
        analyzer = LogAnalyzer(log_content)
        entries = analyzer.extract_entries()
        
        # Check OnePhonon initialization parameters
        init_entry = next(e for e in entries if e[0] == "OnePhonon.__init__")
        assert "tests/pdbs/5zck.pdb" in init_entry[1]
        assert "[-4, 4, 1]" in init_entry[1]
        assert "[-17, 17, 1]" in init_entry[1]
        assert "[-29, 29, 1]" in init_entry[1]
        assert "'gnm_cutoff': 4.0" in init_entry[1]
        assert "'gamma_intra': 1.0" in init_entry[1]
        assert "'gamma_inter': 1.0" in init_entry[1]
        assert "'expand_p1': True" in init_entry[1]

    def test_log_sequence(self, base_log_file):
        with open(base_log_file) as f:
            log_content = f.read()
        analyzer = LogAnalyzer(log_content)
        expected_sequence = ["ModelRunner.run_model"]
        assert analyzer.validate_sequence(expected_sequence)

    def test_sym_ops_return_value(self):
        import os
        import numpy as np
        import numpy.testing as npt
        # Load the base run log file
        log_file = os.path.join("tests", "test_data", "logs", "base_run", "base_run.log")
        with open(log_file, "r") as f:
            lines = f.readlines()
        # Find the return value entry and capture the complete multi-line string
        ret_lines = []
        capture = False
        for line in lines:
            if line.startswith("[AtomicModel._get_sym_ops] Return value:"):
                capture = True
                _, val = line.split("Return value:", 1)
                ret_lines.append(val.strip())
            elif capture:
                if line.startswith("["):
                    break
                ret_lines.append(line.strip())
        
        assert ret_lines, "No return value entry found in log"
        # Join the captured lines preserving newlines and extract the complete tuple string
        print("AGGRESSIVE DEBUG: ret_lines (list) =", ret_lines)
        ret_val_str = "\n".join(ret_lines)
        print("AGGRESSIVE DEBUG: ret_val_str (raw) =", ret_val_str)
        import re
        # Use regex to match the outermost parentheses of the tuple
        m = re.search(r'^\((.*\}\))\)$', ret_val_str, re.DOTALL)
        if m:
            ret_val_str_clean = m.group(0)
        else:
            ret_val_str_clean = ret_val_str
        print("DEBUG (modified):", ret_val_str_clean)
        # Replace 'array(' with 'np.array(' in the cleaned string
        ret_val_str_mod = re.sub(r'\barray\(', 'np.array(', ret_val_str_clean)
        # Remove any stray newlines
        print("AGGRESSIVE DEBUG: ret_val_str_mod =", ret_val_str_mod)
        print("AGGRESSIVE DEBUG: repr(ret_val_str_mod) =", repr(ret_val_str_mod))
        ret_val_str_mod = ret_val_str_mod.replace("\n", " ")
        print("DEBUG: ret_lines =", ret_lines)
        print("DEBUG: ret_val_str =", ret_val_str)
        print("DEBUG: ret_val_str_mod =", ret_val_str_mod)
    
        globals_dict = {"np": np}
        print("AGGRESSIVE DEBUG: globals_dict =", globals_dict)
        print("AGGRESSIVE DEBUG: type(ret_val_str_mod) =", type(ret_val_str_mod))
        try:
            ret_val = eval(ret_val_str_mod, globals_dict)
        except Exception as e:
            print(f"Failed to eval string: {ret_val_str_mod}")
            raise
        # Expected value tuple for the log
        expected_sym_ops = (
            {
                0: np.array([[1., 0., 0.],
                             [0., 1., 0.],
                             [0., 0., 1.]]),
                1: np.array([[-1., 0., 0.],
                             [0., -1., 0.],
                             [0., 0., 1.]]),
                2: np.array([[-1., 0., 0.],
                             [0., 1., 0.],
                             [0., 0., -1.]]),
                3: np.array([[1., 0., 0.],
                             [0., -1., 0.],
                             [0., 0., -1.]])
            },
            {
                0: np.array([[1., 0., 0., 0.],
                             [0., 1., 0., 0.],
                             [0., 0., 1., 0.]]),
                1: np.array([[-1.0, 0.0, 0.0, 2.4065],
                             [0.0, -1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 14.782]]),
                2: np.array([[-1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 8.5755],
                             [0.0, 0.0, -1.0, 14.782]]),
                3: np.array([[1.0, 0.0, 0.0, 2.4065],
                             [0.0, -1.0, 0.0, 8.5755],
                             [0.0, 0.0, -1.0, 0.0]])
            }
        )
        # Compare the dictionaries element-wise
        for key in expected_sym_ops[0]:
            npt.assert_array_almost_equal(ret_val[0][key], expected_sym_ops[0][key])
        for key in expected_sym_ops[1]:
            npt.assert_array_almost_equal(ret_val[1][key], expected_sym_ops[1][key])

    def test_failure_message_present(self, base_log_file):
        with open(base_log_file) as f:
            log_content = f.read()
        analyzer = LogAnalyzer(log_content)
        failure_msg = analyzer.get_failure_message()
        assert failure_msg is None

    def test_edge_case_invalid_gamma(self, edge_log_file):
        """Test behavior with invalid gamma_inter parameter"""
        with open(edge_log_file) as f:
            log_content = f.read()
        analyzer = LogAnalyzer(log_content)
        failure_msg = analyzer.get_failure_message()
        assert failure_msg is not None
        expected_sequence = ["ModelRunner.run_model"]
        assert analyzer.validate_sequence(expected_sequence)

    def test_computation_validation(self):
        """Test that computed diffraction pattern matches reference data"""
        import numpy as np
        from eryx.models import OnePhonon
        onephonon = OnePhonon("tests/pdbs/5zck_p1.pdb", [-4,4,3], [-17,17,3], [-29,29,3],
                               expand_p1=True, gnm_cutoff=4.0, gamma_intra=1.0, gamma_inter=1.0)
        computed = onephonon.apply_disorder().flatten()
        ref = np.load("tests/test_data/reference/diffraction_pattern.npy")
        valid = ~np.isnan(ref)
        import numpy as np
        np.testing.assert_allclose(computed[valid], ref[valid], rtol=1e-2)
