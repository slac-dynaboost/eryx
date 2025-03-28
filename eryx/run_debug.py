# This is a symlink to the root run_debug.py file
# to make it accessible as eryx.run_debug
from .. import run_debug

# Re-export the functions
run_np = run_debug.run_np
run_torch = run_debug.run_torch
setup_logging = run_debug.setup_logging
