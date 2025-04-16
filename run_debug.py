#!/usr/bin/env python3
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from eryx.models import OnePhonon

# Your existing setup_logging function remains the same
def setup_logging():
    # Remove any existing handlers.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        filename="debug_output.log",
        filemode="w"
    )
    # Also output to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

def run_np():
    # Use a small grid for testing; adjust parameters as necessary.
    logging.info("Starting NP branch computation")
    pdb_path = "tests/pdbs/5zck_p1.pdb"
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
    
    # --- Save q-vectors and intensity to NPZ ---
    q_vectors_np = onephonon_np.q_grid
    intensity_np = Id_np

    output_filename = "np_results.npz"
    try:
        np.savez_compressed(output_filename,
                            q_vectors=q_vectors_np,
                            intensity=intensity_np,
                            map_shape=onephonon_np.map_shape)  # Save map_shape too
        logging.info(f"NP: Saved q-vectors ({q_vectors_np.shape}) and intensity ({intensity_np.shape}) to {output_filename}")
        logging.info("NP branch diffuse intensity stats: min=%s, max=%s", np.nanmin(intensity_np), np.nanmax(intensity_np))
    except Exception as e:
        logging.error(f"NP: Failed to save results to {output_filename}: {e}")
    
    # Also save the original .npy file for backward compatibility
    np.save("np_diffuse_intensity.npy", Id_np)


def run_torch():
    """Run PyTorch version of the diffuse scattering simulation."""
    try:
        import torch
        from eryx.models_torch import OnePhonon
        
        # Get the device (use CUDA if available)
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        logging.info(f"Starting PyTorch branch computation on {device}")
        
        # Use the same parameters as in run_np
        pdb_path = "tests/pdbs/5zck_p1.pdb"
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
        
        # Apply disorder
        Id_torch = onephonon_torch.apply_disorder(use_data_adp=True)
        
        # Log debug information
        logging.debug(f"PyTorch: hkl_grid shape = {onephonon_torch.hkl_grid.shape}")
        logging.debug("PyTorch: hkl_grid coordinate ranges:")
        logging.debug(f"  Dimension 0: min = {onephonon_torch.hkl_grid[:,0].min().item()}, max = {onephonon_torch.hkl_grid[:,0].max().item()}")
        logging.debug(f"  Dimension 1: min = {onephonon_torch.hkl_grid[:,1].min().item()}, max = {onephonon_torch.hkl_grid[:,1].max().item()}")
        logging.debug(f"  Dimension 2: min = {onephonon_torch.hkl_grid[:,2].min().item()}, max = {onephonon_torch.hkl_grid[:,2].max().item()}")
        logging.debug(f"PyTorch: q_grid range: min = {onephonon_torch.q_grid.min().item()}, max = {onephonon_torch.q_grid.max().item()}")
        
        # --- Save q-vectors and intensity to NPZ ---
        q_vectors_torch = onephonon_torch.q_grid.detach().cpu().numpy()
        intensity_torch_np = Id_torch.detach().cpu().numpy()

        output_filename = "torch_grid_results.npz"
        try:
            np.savez_compressed(output_filename,
                                q_vectors=q_vectors_torch,
                                intensity=intensity_torch_np,
                                map_shape=onephonon_torch.map_shape)  # Save map_shape too
            logging.info(f"PyTorch Grid: Saved q-vectors ({q_vectors_torch.shape}) and intensity ({intensity_torch_np.shape}) to {output_filename}")
            logging.info("PyTorch Grid diffuse intensity stats: min=%s, max=%s", 
                         np.nanmin(intensity_torch_np), np.nanmax(intensity_torch_np))
        except Exception as e:
            logging.error(f"PyTorch Grid: Failed to save results to {output_filename}: {e}")
        
        # Also save original files for backward compatibility
        torch.save(Id_torch, "torch_diffuse_intensity.pt")
        np.save("torch_diffuse_intensity.npy", Id_torch.detach().cpu().numpy())
        
        return Id_torch
        
    except RuntimeError as e:
        # Handle CUDA errors by falling back to CPU
        if 'CUDA' in str(e):
            logging.error(f"CUDA error: {e}. Attempting to run on CPU instead.")
            # Modify the environment to force CPU usage and retry
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return run_torch()  # Recursive call will use CPU now
        else:
            logging.error(f"Error in PyTorch computation: {e}")
            raise
    except Exception as e:
        logging.error(f"Unexpected error in PyTorch computation: {e}")

def run_torch_arb_q():
    """Run PyTorch version in arbitrary q-vector mode."""
    try:
        import torch
        from eryx.models_torch import OnePhonon
        
        # Get the device (use CPU for consistency with other runs)
        device = torch.device('cpu')
        logging.info(f"Starting PyTorch arbitrary q-vector mode computation on {device}")
        
        # First, run grid mode to extract q vectors
        logging.info("Creating grid model to extract q-vectors...")
        pdb_path = "tests/pdbs/5zck_p1.pdb"
        grid_model = OnePhonon(
            pdb_path,
            [-4, 4, 3], [-17, 17, 3], [-29, 29, 3],
            expand_p1=True,
            res_limit=0.0,
            gnm_cutoff=4.0,
            gamma_intra=1.0,
            gamma_inter=1.0,
            device=device
        )
        
        # Extract q-vectors and map shape
        q_vectors = grid_model.q_grid.clone().detach()
        map_shape = grid_model.map_shape
        hkl_grid = grid_model.hkl_grid.clone().detach()
        logging.info(f"Extracted q-vectors shape: {q_vectors.shape}")
        logging.info(f"Map shape: {map_shape}")
        
        # Save memory by deleting the grid model
        del grid_model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        # Now create the arbitrary q-vector model
        logging.info("Creating arbitrary q-vector model...")
        # Pass sampling parameters explicitly as they are required for ADP calculation in GNM model
        hsampling_params = [-4, 4, 3]
        ksampling_params = [-17, 17, 3]
        lsampling_params = [-29, 29, 3]
        arb_q_model = OnePhonon(
            pdb_path,
            q_vectors=q_vectors,
            hsampling=hsampling_params, # Pass sampling params
            ksampling=ksampling_params,
            lsampling=lsampling_params,
            expand_p1=True,
            res_limit=0.0,
            gnm_cutoff=4.0,
            gamma_intra=1.0,
            gamma_inter=1.0,
            device=device
        )
        logging.info("Arbitrary q-vector model initialized (phonons/ADPs calculated automatically).")

        # Verify phonon tensors exist and contain non-zero values
        if hasattr(arb_q_model, 'V') and hasattr(arb_q_model, 'Winv') and arb_q_model.V is not None and arb_q_model.Winv is not None:
            v_nonzero = torch.count_nonzero(torch.abs(arb_q_model.V)).item()
            winv_nonzero = torch.count_nonzero(~torch.isnan(arb_q_model.Winv)).item() if arb_q_model.Winv is not None else 0
            
            logging.info(f"Phonon eigenvectors (V) shape: {arb_q_model.V.shape}")
            logging.info(f"Phonon eigenvalues (Winv) shape: {arb_q_model.Winv.shape}")
            logging.info(f"Non-zero elements - V: {v_nonzero}, Winv: {winv_nonzero}")
        else:
            logging.warning("Phonon tensors (V and/or Winv) are missing.")
        
        # Apply disorder to get intensity
        logging.info("Computing diffuse intensity...")
        Id_arb_q = arb_q_model.apply_disorder(use_data_adp=True)
        
        # Log stats and save results
        valid_intensity = Id_arb_q[~torch.isnan(Id_arb_q)]
        if valid_intensity.numel() > 0:
            min_intensity = valid_intensity.min().item()
            max_intensity = valid_intensity.max().item()
            mean_intensity = valid_intensity.mean().item()
            logging.info(f"Arb-Q diffuse intensity stats: min={min_intensity}, max={max_intensity}, mean={mean_intensity}")
        else:
            logging.warning("All intensity values are NaN!")
        
        # --- Save q-vectors and intensity to NPZ ---
        q_vectors_arbq_np = q_vectors.detach().cpu().numpy()
        intensity_arbq_np = Id_arb_q.detach().cpu().numpy()

        output_filename = "torch_arbq_results.npz"
        try:
            np.savez_compressed(output_filename,
                                q_vectors=q_vectors_arbq_np,
                                intensity=intensity_arbq_np,
                                map_shape=map_shape)  # Save the map_shape derived from grid
            logging.info(f"Arb-Q: Saved q-vectors ({q_vectors_arbq_np.shape}) and intensity ({intensity_arbq_np.shape}) to {output_filename}")
            
            # Log stats from flat data
            valid_intensity_np = intensity_arbq_np[~np.isnan(intensity_arbq_np)]
            if valid_intensity_np.size > 0:
                min_intensity = np.min(valid_intensity_np)
                max_intensity = np.max(valid_intensity_np)
                mean_intensity = np.mean(valid_intensity_np)
                logging.info(f"Arb-Q flat intensity stats: min={min_intensity:.4f}, max={max_intensity:.4f}, mean={mean_intensity:.4f}")
            else:
                logging.warning("Arb-Q: All flat intensity values are NaN!")
        except Exception as e:
            logging.error(f"Arb-Q: Failed to save results to {output_filename}: {e}")
        
        # Also save original files for backward compatibility
        torch.save(Id_arb_q, "arb_q_diffuse_intensity.pt")
        np.save("arb_q_diffuse_intensity.npy", Id_arb_q.detach().cpu().numpy())
        
        return Id_arb_q, map_shape
        
    except Exception as e:
        logging.error(f"Error in arbitrary q-vector mode: {e}", exc_info=True)
        raise

def visualize_results():
    """Visualize and compare the results from all three methods."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Define NPZ filenames
        np_file = "np_results.npz"
        torch_file = "torch_grid_results.npz"
        arb_q_file = "torch_arbq_results.npz"
        
        files_exist = all(os.path.exists(f) for f in [np_file, torch_file, arb_q_file])
        if not files_exist:
            missing = [f for f in [np_file, torch_file, arb_q_file] if not os.path.exists(f)]
            logging.warning(f"Not all result files exist. Missing: {', '.join(missing)}. Please run all methods first.")
            # Fall back to old files if available
            old_np_file = "np_diffuse_intensity.npy"
            old_torch_file = "torch_diffuse_intensity.npy"
            old_arb_q_file = "arb_q_diffuse_intensity.npy"
            if all(os.path.exists(f) for f in [old_np_file, old_torch_file, old_arb_q_file]):
                logging.info("Found old .npy files, using those instead.")
                Id_np = np.load(old_np_file)
                Id_torch = np.load(old_torch_file)
                Id_arb_q = np.load(old_arb_q_file)
                
                # Get the map shape
                map_shape = Id_np.shape
                logging.info(f"Visualization: Map shape = {map_shape}")
                
                # Check if Id_arb_q needs reshaping
                if Id_arb_q.ndim == 1:
                    logging.info(f"Reshaping arbitrary q-vector results from 1D to 3D ({Id_arb_q.shape} -> {map_shape})")
                    Id_arb_q = Id_arb_q.reshape(map_shape)
            else:
                return
        else:
            # Load data from NPZ files
            data_np = np.load(np_file)
            Id_np = data_np['intensity']
            q_np = data_np['q_vectors']
            map_shape_np = tuple(data_np['map_shape']) if 'map_shape' in data_np else None
            
            data_torch = np.load(torch_file)
            Id_torch = data_torch['intensity']
            q_torch = data_torch['q_vectors']
            map_shape_torch = tuple(data_torch['map_shape']) if 'map_shape' in data_torch else None
            
            data_arbq = np.load(arb_q_file)
            Id_arb_q = data_arbq['intensity']
            q_arbq = data_arbq['q_vectors']
            map_shape_arbq = tuple(data_arbq['map_shape']) if 'map_shape' in data_arbq else None
            
            # Determine consistent map_shape
            map_shape = map_shape_np or map_shape_torch or map_shape_arbq
            if map_shape is None:
                logging.error("Could not determine map_shape from any NPZ file.")
                return
            
            logging.info(f"Using map_shape: {map_shape}")
            
            # Reshape if needed
            if Id_np.ndim == 1 and Id_np.size == np.prod(map_shape):
                Id_np = Id_np.reshape(map_shape)
            if Id_torch.ndim == 1 and Id_torch.size == np.prod(map_shape):
                Id_torch = Id_torch.reshape(map_shape)
            if Id_arb_q.ndim == 1 and Id_arb_q.size == np.prod(map_shape):
                Id_arb_q = Id_arb_q.reshape(map_shape)
        
        # Create figure for central slices
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('Comparison of Diffuse Intensity Central Slices', fontsize=16)
        
        # Determine reasonable color limits, ignoring NaNs
        all_data = np.concatenate([
            Id_np[~np.isnan(Id_np)].flatten(),
            Id_torch[~np.isnan(Id_torch)].flatten(),
            Id_arb_q[~np.isnan(Id_arb_q)].flatten()
        ])
        
        if len(all_data) > 0:
            vmin = np.percentile(all_data, 1)
            vmax = np.percentile(all_data, 99)
            if vmin == vmax:
                vmin = np.min(all_data) if np.min(all_data) < vmax else vmax * 0.9
                vmax = np.max(all_data) if np.max(all_data) > vmin else vmin * 1.1
        else:
            vmin, vmax = 0, 1
        
        # Central indices
        h_mid = map_shape[0] // 2
        k_mid = map_shape[1] // 2
        l_mid = map_shape[2] // 2
        
        # Helper function to plot a slice
        def plot_slice(ax, data, slice_idx, slice_dim, title):
            if slice_dim == 0:
                im = ax.imshow(data[slice_idx, :, :], origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
                ax.set_xlabel('l index')
                ax.set_ylabel('k index')
            elif slice_dim == 1:
                im = ax.imshow(data[:, slice_idx, :], origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
                ax.set_xlabel('l index')
                ax.set_ylabel('h index')
            else:
                im = ax.imshow(data[:, :, slice_idx], origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
                ax.set_xlabel('k index')
                ax.set_ylabel('h index')
            ax.set_title(title)
            return im
        
        # Plot h, k, l slices for all three methods
        im1 = plot_slice(axes[0, 0], Id_np, h_mid, 0, 'NumPy (h slice)')
        im2 = plot_slice(axes[0, 1], Id_torch, h_mid, 0, 'PyTorch Grid (h slice)')
        im3 = plot_slice(axes[0, 2], Id_arb_q, h_mid, 0, 'PyTorch Arb-Q (h slice)')
        
        plot_slice(axes[1, 0], Id_np, k_mid, 1, 'NumPy (k slice)')
        plot_slice(axes[1, 1], Id_torch, k_mid, 1, 'PyTorch Grid (k slice)')
        plot_slice(axes[1, 2], Id_arb_q, k_mid, 1, 'PyTorch Arb-Q (k slice)')
        
        plot_slice(axes[2, 0], Id_np, l_mid, 2, 'NumPy (l slice)')
        plot_slice(axes[2, 1], Id_torch, l_mid, 2, 'PyTorch Grid (l slice)')
        plot_slice(axes[2, 2], Id_arb_q, l_mid, 2, 'PyTorch Arb-Q (l slice)')
        
        # Add colorbar
        cbar = fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Intensity')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('diffuse_intensity_comparison.png', dpi=300)
        logging.info("Saved slice comparison to diffuse_intensity_comparison.png")
        plt.show()
        
        # Create figure for histogram comparison
        plt.figure(figsize=(12, 6))
        
        # Filter out NaNs and extreme values for better visualization
        def filter_data_for_hist(data):
            valid_data = data[~np.isnan(data)].flatten()
            if len(valid_data) > 0:
                p999 = np.percentile(valid_data, 99.9)
                return valid_data[valid_data <= p999]
            return []
        
        np_hist_data = filter_data_for_hist(Id_np)
        torch_hist_data = filter_data_for_hist(Id_torch)
        arb_q_hist_data = filter_data_for_hist(Id_arb_q)
        
        # Plot histograms
        plt.hist(np_hist_data, bins=50, alpha=0.5, label='NumPy')
        plt.hist(torch_hist_data, bins=50, alpha=0.5, label='PyTorch Grid')
        plt.hist(arb_q_hist_data, bins=50, alpha=0.5, label='PyTorch Arb-Q')
        
        plt.title('Histogram of Diffuse Intensity Values')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('diffuse_intensity_histogram.png', dpi=300)
        logging.info("Saved histogram comparison to diffuse_intensity_histogram.png")
        plt.show()
        
        # Calculate and log statistics for comparison
        logging.info("\n===== Intensity Statistics Comparison =====")
        for name, data in [("NumPy", Id_np), ("PyTorch Grid", Id_torch), ("PyTorch Arb-Q", Id_arb_q)]:
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                stats = {
                    "min": np.min(valid_data),
                    "max": np.max(valid_data),
                    "mean": np.mean(valid_data),
                    "median": np.median(valid_data),
                    "std": np.std(valid_data),
                    "valid_points": len(valid_data),
                    "nan_points": np.sum(np.isnan(data))
                }
                logging.info(f"{name} statistics:")
                for key, value in stats.items():
                    logging.info(f"  {key}: {value}")
            else:
                logging.info(f"{name}: No valid data points (all NaN)")
        
    except Exception as e:
        logging.error(f"Error during visualization: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    setup_logging()
    
    # Run all three methods
    run_np()
    run_torch()
    run_torch_arb_q()
    
    # Visualize the results
    visualize_results()
    
    logging.info("Completed debug run with visualization. Check diffuse_intensity_comparison.png and diffuse_intensity_histogram.png")
