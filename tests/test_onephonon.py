import pytest
import numpy as np
from tests.test_utils.log_analysis import LogAnalyzer

class TestOnePhonon:

    @pytest.fixture
    def base_log_content(self):
        # Simulated base log content from a successful run (except the final failure message)
        return (
            "[OnePhonon.__init__] Enter with args=('tests/pdbs/5zck.pdb', [-4, 4, 1], [-17, 17, 1], [-29, 29, 1]), kwargs={'gnm_cutoff': 4.0, 'gamma_intra': 1.0, 'gamma_inter': 1.0, 'expand_p1': True}\n"
            "[OnePhonon._setup] Enter with args=('tests/pdbs/5zck.pdb', True, 0.0, 'asu'), kwargs={}\n"
            "[AtomicModel.__init__] Enter with args=('tests/pdbs/5zck.pdb', True), kwargs={}\n"
            "[AtomicModel.extract_frame] Enter with args=(), kwargs={'frame': 0, 'expand_p1': True}\n"
            "Model run failed: 'NoneType' object has no attribute 'shape'"
        )

    @pytest.fixture
    def detailed_log_content(self):
        # Simulated log content containing a duration value entry.
        return (
            "[AtomicModel._get_gemmi_structure] Enter with args=('tests/pdbs/5zck.pdb', True), kwargs={}\n"
            "[AtomicModel._get_gemmi_structure] Exit. Duration=0.12s\n"
            "[AtomicModel.extract_frame] Enter with args=(), kwargs={'frame': 0, 'expand_p1': True}\n"
        )

    @pytest.fixture
    def edge_log_content(self):
        # Simulated edge case log content (with an error condition: invalid gamma_inter)
        return (
            "[OnePhonon.__init__] Enter with args=('tests/pdbs/5zck.pdb', [-4, 4, 1], [-17, 17, 1], [-29, 29, 1]), kwargs={'gnm_cutoff': 4.0, 'gamma_intra': 1.0, 'gamma_inter': -1.0, 'expand_p1': True}\n"
            "[OnePhonon._setup] Enter with args=('tests/pdbs/5zck.pdb', True, 0.0, 'asu'), kwargs={}\n"
            "[AtomicModel.__init__] Enter with args=('tests/pdbs/5zck.pdb', True), kwargs={}\n"
            "[AtomicModel.extract_frame] Enter with args=(), kwargs={'frame': 0, 'expand_p1': True}\n"
            "Model run failed: 'NoneType' object has no attribute 'shape'"
        )

    def test_log_sequence(self, base_log_content):
        analyzer = LogAnalyzer(base_log_content)
        expected_sequence = [
            "OnePhonon.__init__",
            "OnePhonon._setup",
            "AtomicModel.__init__",
            "AtomicModel.extract_frame"
        ]
        assert analyzer.validate_sequence(expected_sequence)

    def test_failure_message_present(self, base_log_content):
        analyzer = LogAnalyzer(base_log_content)
        failure_msg = analyzer.get_failure_message()
        assert failure_msg is not None
        assert "Model run failed" in failure_msg

    def test_get_duration_absent(self, base_log_content):
        # With the base_log_content fixture, no duration is present.
        analyzer = LogAnalyzer(base_log_content)
        duration = analyzer.get_duration("AtomicModel._get_gemmi_structure")
        assert duration is None

    def test_get_duration_detailed(self, detailed_log_content):
        analyzer = LogAnalyzer(detailed_log_content)
        duration = analyzer.get_duration("AtomicModel._get_gemmi_structure")
        assert duration == 0.12

    def test_integration_edge_case(self, edge_log_content):
        analyzer = LogAnalyzer(edge_log_content)
        failure_msg = analyzer.get_failure_message()
        assert failure_msg is not None
        expected_sequence = [
            "OnePhonon.__init__",
            "OnePhonon._setup",
            "AtomicModel.__init__",
            "AtomicModel.extract_frame"
        ]
        assert analyzer.validate_sequence(expected_sequence)
