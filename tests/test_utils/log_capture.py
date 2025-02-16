import logging

class LogCapture(logging.Handler):
    """
    Custom logging handler to capture logs in memory.
    """
    def __init__(self):
        super().__init__()
        self.logs = []
        
    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)
    
    def get_logs(self):
        return self.logs

class LogStore:
    """
    Manages log persistence and retrieval.
    """
    def __init__(self, storage_path):
        self.storage_path = storage_path
        
    def save_logs(self, logs):
        with open(self.storage_path, 'w') as f:
            for log in logs:
                f.write(log + "\n")
                
    def load_logs(self):
        with open(self.storage_path, 'r') as f:
            return f.read().splitlines()
