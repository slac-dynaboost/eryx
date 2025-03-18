#!/usr/bin/env python3
"""
PyTorch implementation of diffuse scattering simulation.

This script provides a PyTorch-based implementation of the diffuse scattering
simulation, parallel to the NumPy implementation in run_debug.py. It enables
gradient-based optimization of simulation parameters.
"""

import os
import logging
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any

# Set default PyTorch dtype for consistency across the application
torch.set_default_dtype(torch.float32)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    filename="torch_debug_output.log",
    filemode="w"
)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logging.getLogger("").addHandler(console)

def setup_logging():
    """
    Set up logging configuration.
    """
    # Remove any existing handlers.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        filename="torch_debug_output.log",
        filemode="w"
    )
    # Also output to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

def run_torch(device: Optional[torch.device] = None):
    """
    Run PyTorch version of the diffuse scattering simulation.
    
    Args:
        device: PyTorch device to use (default: CUDA if available, else CPU)
        
    Returns:
        torch.Tensor: Diffuse intensity tensor
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        logging.info(f"Starting PyTorch branch computation on {device}")
        
        # Import the PyTorch OnePhonon implementation
        from eryx.models_torch import OnePhonon
        
        # Same parameters as NumPy version
        pdb_path = "tests/pdbs/5zck_p1.pdb"
        
        # Create OnePhonon instance with optimized parameters
        onephonon_torch = OnePhonon(
            pdb_path,
            [-4, 4, 3], [-17, 17, 3], [-29, 29, 3],
            expand_p1=True,
            res_limit=0.0,
            gnm_cutoff=4.0,
            gamma_intra=1.0,
            gamma_inter=1.0,
            device=device
        )
        
        # Apply disorder model
        Id_torch = onephonon_torch.apply_disorder(use_data_adp=True)
        
        # Log debug information
        logging.debug(f"PyTorch: hkl_grid shape = {onephonon_torch.hkl_grid.shape}")
        logging.debug("PyTorch: hkl_grid coordinate ranges:")
        logging.debug(f"  Dimension 0: min = {onephonon_torch.hkl_grid[:,0].min().item()}, max = {onephonon_torch.hkl_grid[:,0].max().item()}")
        logging.debug(f"  Dimension 1: min = {onephonon_torch.hkl_grid[:,1].min().item()}, max = {onephonon_torch.hkl_grid[:,1].max().item()}")
        logging.debug(f"  Dimension 2: min = {onephonon_torch.hkl_grid[:,2].min().item()}, max = {onephonon_torch.hkl_grid[:,2].max().item()}")
        logging.debug(f"PyTorch: q_grid range: min = {onephonon_torch.q_grid.min().item()}, max = {onephonon_torch.q_grid.max().item()}")
        logging.info("PyTorch branch diffuse intensity stats: min=%s, max=%s", 
                    torch.nanmin(Id_torch).item(), torch.nanmax(Id_torch).item())
        
        # Save results for comparison
        torch.save(Id_torch, "torch_diffuse_intensity.pt")
        np.save("torch_diffuse_intensity.npy", Id_torch.detach().cpu().numpy())
        
        return Id_torch
        
    except RuntimeError as e:
        if device.type == 'cuda' and 'CUDA' in str(e):
            logging.error(f"CUDA error: {e}. Falling back to CPU.")
            return run_torch(device=torch.device('cpu'))
        else:
            logging.error(f"Error in PyTorch computation: {e}")
            raise

def run_np():
    """
    Run NumPy version of the diffuse scattering simulation.
    
    This is the same function as in run_debug.py, included for direct comparison.
    """
    logging.info("Starting NP branch computation")
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    
    from eryx.models import OnePhonon
    
    onephonon_np = OnePhonon(
        pdb_path,
        [-4, 4, 3], [-17, 17, 3], [-29, 29, 3],
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0
    )
    Id_np = onephonon_np.apply_disorder(use_data_adp=True)
    logging.debug(f"NP: hkl_grid shape = {onephonon_np.hkl_grid.shape}")
    logging.debug("NP: hkl_grid coordinate ranges:")
    logging.debug(f"  Dimension 0: min = {onephonon_np.hkl_grid[:,0].min()}, max = {onephonon_np.hkl_grid[:,0].max()}")
    logging.debug(f"  Dimension 1: min = {onephonon_np.hkl_grid[:,1].min()}, max = {onephonon_np.hkl_grid[:,1].max()}")
    logging.debug(f"  Dimension 2: min = {onephonon_np.hkl_grid[:,2].min()}, max = {onephonon_np.hkl_grid[:,2].max()}")
    logging.debug(f"NP: q_grid range: min = {onephonon_np.q_grid.min()}, max = {onephonon_np.q_grid.max()}")
    logging.info("NP branch diffuse intensity stats: min=%s, max=%s", np.nanmin(Id_np), np.nanmax(Id_np))
    np.save("np_diffuse_intensity.npy", Id_np)

def compare_results(rtol: float = 1e-5, atol: float = 1e-8) -> Dict[str, float]:
    """
    Compare the results from NumPy and PyTorch implementations.
    
    Args:
        rtol: Relative tolerance for numerical comparison
        atol: Absolute tolerance for numerical comparison
        
    Returns:
        Dictionary with comparison metrics
    """
    logging.info("Comparing NumPy and PyTorch results...")
    
    # Load results
    try:
        np_result = np.load("np_diffuse_intensity.npy")
        torch_result = np.load("torch_diffuse_intensity.npy")
    except FileNotFoundError as e:
        logging.error(f"Could not load result files: {e}")
        raise
    
    # Create mask for non-NaN values in both arrays
    mask = ~np.isnan(np_result) & ~np.isnan(torch_result)
    if not np.any(mask):
        logging.warning("All values are NaN in at least one of the arrays")
        return {
            'valid_points': 0,
            'mse': float('nan'),
            'rmse': float('nan'),
            'correlation': float('nan'),
            'max_abs_diff': float('nan'),
            'mean_abs_diff': float('nan'),
            'relative_diff_percent': float('nan'),
            'within_tolerance': False
        }
    
    # Compute statistics
    valid_points = np.sum(mask)
    mse = np.nanmean((np_result[mask] - torch_result[mask])**2)
    rmse = np.sqrt(mse)
    
    # Compute correlation
    correlation = np.corrcoef(np_result[mask], torch_result[mask])[0, 1]
    
    # Compute detailed statistics
    abs_diff = np.abs(np_result[mask] - torch_result[mask])
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    # Compute relative difference
    # Avoid division by zero
    denominator = np.maximum(np.abs(np_result[mask]), 1e-10)
    relative_diff = np.mean(abs_diff / denominator) * 100  # as percentage
    
    # Check if within tolerance
    within_tolerance = np.allclose(np_result[mask], torch_result[mask], rtol=rtol, atol=atol)
    
    # Create metrics dictionary
    metrics = {
        'valid_points': valid_points,
        'mse': mse,
        'rmse': rmse,
        'correlation': correlation,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'relative_diff_percent': relative_diff,
        'within_tolerance': within_tolerance
    }
    
    # Log comparison results
    logging.info(f"Comparison results (rtol={rtol}, atol={atol}):")
    logging.info(f"  Valid points: {valid_points} out of {np_result.size}")
    logging.info(f"  MSE: {mse:.8e}")
    logging.info(f"  RMSE: {rmse:.8e}")
    logging.info(f"  Correlation: {correlation:.6f}")
    logging.info(f"  Max absolute difference: {max_abs_diff:.8e}")
    logging.info(f"  Mean absolute difference: {mean_abs_diff:.8e}")
    logging.info(f"  Mean relative difference: {relative_diff:.4f}%")
    logging.info(f"  Within tolerance: {within_tolerance}")
    
    return metrics

def benchmark_performance(pdb_path: str, runs: int = 3) -> Dict[str, Dict[str, float]]:
    """
    Benchmark NumPy vs PyTorch implementations.
    
    Args:
        pdb_path: Path to PDB file
        runs: Number of runs for averaging performance
        
    Returns:
        Dictionary with performance metrics
    """
    import time
    import gc  # for garbage collection
    
    logging.info(f"Benchmarking performance with {runs} runs...")
    
    # Define common parameters
    params = {
        'pdb_path': pdb_path,
        'hsampling': [-4, 4, 3],
        'ksampling': [-17, 17, 3],
        'lsampling': [-29, 29, 3],
        'expand_p1': True,
        'res_limit': 0.0,
        'gnm_cutoff': 4.0,
        'gamma_intra': 1.0,
        'gamma_inter': 1.0
    }
    
    # Initialize metrics dictionary
    metrics = {
        'numpy': {'time': 0.0},
        'pytorch_cpu': {'time': 0.0},
        'pytorch_gpu': {'time': 0.0, 'memory': 0.0}
    }
    
    # NumPy benchmark
    logging.info("Benchmarking NumPy implementation...")
    from eryx.models import OnePhonon as NumpyOnePhonon
    
    np_times = []
    for i in range(runs):
        gc.collect()  # Force garbage collection
        start_time = time.time()
        
        model_np = NumpyOnePhonon(**params)
        Id_np = model_np.apply_disorder(use_data_adp=True)
        
        end_time = time.time()
        np_times.append(end_time - start_time)
        logging.info(f"  Run {i+1}/{runs}: {np_times[-1]:.2f} seconds")
    
    metrics['numpy']['time'] = sum(np_times) / len(np_times)
    logging.info(f"NumPy average time: {metrics['numpy']['time']:.2f} seconds")
    
    # PyTorch CPU benchmark
    logging.info("Benchmarking PyTorch CPU implementation...")
    from eryx.models_torch import OnePhonon as TorchOnePhonon
    
    cpu_times = []
    for i in range(runs):
        gc.collect()  # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache
        
        start_time = time.time()
        
        model_torch_cpu = TorchOnePhonon(**params, device=torch.device('cpu'))
        Id_torch_cpu = model_torch_cpu.apply_disorder(use_data_adp=True)
        
        end_time = time.time()
        cpu_times.append(end_time - start_time)
        logging.info(f"  Run {i+1}/{runs}: {cpu_times[-1]:.2f} seconds")
    
    metrics['pytorch_cpu']['time'] = sum(cpu_times) / len(cpu_times)
    logging.info(f"PyTorch CPU average time: {metrics['pytorch_cpu']['time']:.2f} seconds")
    
    # PyTorch GPU benchmark (if available)
    if torch.cuda.is_available():
        logging.info("Benchmarking PyTorch GPU implementation...")
        
        gpu_times = []
        gpu_memory = []
        
        for i in range(runs):
            gc.collect()  # Force garbage collection
            torch.cuda.empty_cache()  # Clear CUDA cache
            torch.cuda.reset_peak_memory_stats()  # Reset memory stats
            
            start_time = time.time()
            
            model_torch_gpu = TorchOnePhonon(**params, device=torch.device('cuda'))
            Id_torch_gpu = model_torch_gpu.apply_disorder(use_data_adp=True)
            
            # Synchronize CUDA for accurate timing
            torch.cuda.synchronize()
            
            end_time = time.time()
            gpu_times.append(end_time - start_time)
            
            # Get peak memory usage
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
            gpu_memory.append(memory_used)
            
            logging.info(f"  Run {i+1}/{runs}: {gpu_times[-1]:.2f} seconds, {memory_used:.2f} MB")
            
            # Clean up GPU memory
            del model_torch_gpu, Id_torch_gpu
            torch.cuda.empty_cache()
        
        metrics['pytorch_gpu']['time'] = sum(gpu_times) / len(gpu_times)
        metrics['pytorch_gpu']['memory'] = sum(gpu_memory) / len(gpu_memory)
        
        logging.info(f"PyTorch GPU average time: {metrics['pytorch_gpu']['time']:.2f} seconds")
        logging.info(f"PyTorch GPU average memory: {metrics['pytorch_gpu']['memory']:.2f} MB")
    else:
        logging.info("CUDA not available, skipping GPU benchmark")
    
    # Calculate speedups
    np_time = metrics['numpy']['time']
    cpu_time = metrics['pytorch_cpu']['time']
    cpu_speedup = np_time / cpu_time if cpu_time > 0 else 0
    
    logging.info(f"CPU speedup over NumPy: {cpu_speedup:.2f}x")
    
    if torch.cuda.is_available():
        gpu_time = metrics['pytorch_gpu']['time']
        gpu_speedup = np_time / gpu_time if gpu_time > 0 else 0
        gpu_vs_cpu = cpu_time / gpu_time if gpu_time > 0 else 0
        
        logging.info(f"GPU speedup over NumPy: {gpu_speedup:.2f}x")
        logging.info(f"GPU speedup over CPU PyTorch: {gpu_vs_cpu:.2f}x")
    
    return metrics

if __name__ == "__main__":
    setup_logging()
    run_np()
    Id_torch = run_torch()
    metrics = compare_results()
    
    if metrics['within_tolerance']:
        logging.info("NumPy and PyTorch results match within tolerance!")
    else:
        logging.warning("NumPy and PyTorch results differ beyond tolerance!")
    
    # Uncomment to run benchmarks
    # benchmark_performance("tests/pdbs/5zck_p1.pdb", runs=3)
    
    logging.info("Completed debug run. Please check torch_debug_output.log, np_diffuse_intensity.npy and torch_diffuse_intensity.npy")
