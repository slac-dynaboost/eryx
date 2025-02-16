import time
import logging
from eryx.models import OnePhonon
from tests.test_utils.log_capture import LogCapture

class ModelRunner:
    """
    Runs the OnePhonon model with a given configuration,
    captures logs during execution, and stores run metadata.
    """
    def __init__(self, config):
        self.config = config
        self.log_capture = LogCapture()
        self.logger = logging.getLogger()
        self.logger.addHandler(self.log_capture)
        
    def run_model(self):
        """
        Execute the OnePhonon model.
        Returns a tuple of (result, metadata).
        """
        start_time = time.time()
        try:
            model = OnePhonon(self.config)
            # Assuming the OnePhonon model has a compute_gnm_phonons method
            result = model.compute_gnm_phonons()
            status = "success"
        except Exception as e:
            result = None
            status = "failed"
            self.logger.error(f"Model run failed: {e}")
        end_time = time.time()
        run_time = end_time - start_time
        metadata = {
            "status": status,
            "run_time": run_time,
            "config": self.config
        }
        return result, metadata
        
    def get_captured_logs(self):
        """
        Retrieve captured logs.
        """
        return self.log_capture.get_logs()
