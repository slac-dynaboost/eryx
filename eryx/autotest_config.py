import os
import logging
from eryx.autotest.configuration import Configuration

# Create configuration with debug mode enabled
config = Configuration(debug=True, log_file_prefix="logs")

# Ensure log directory exists
os.makedirs(config.getLogFilePrefix(), exist_ok=True)
