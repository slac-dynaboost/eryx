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
    logging.info("NP branch diffuse intensity stats: min=%s, max=%s", np.nanmin(Id_np), np.nanmax(Id_np))
    # Save for later comparison
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
        
        # Save for later comparison
        torch.save(Id_torch, "torch_diffuse_intensity.pt")
        # Also save as NumPy array for easier comparison
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
        arb_q_model = OnePhonon(
            pdb_path,
            q_vectors=q_vectors,
            hsampling=[-4, 4, 3],  # Still needed for ADP calculation
            ksampling=[-17, 17, 3],
            lsampling=[-29, 29, 3],
            expand_p1=True,
            res_limit=0.0,
            gnm_cutoff=4.0,
            gamma_intra=1.0,
            gamma_inter=1.0,
            device=device
        )
        
        # Explicitly calculate phonon modes (they're skipped in initialization)
        logging.info("Explicitly calculating phonon modes...")
        try:
            # Original calculation
            arb_q_model.compute_gnm_phonons()
            logging.info("Phonon modes calculated successfully")
        except RuntimeError as e:
            if "you can only change requires_grad flags of leaf variables" in str(e):
                logging.warning("Handling gradient issue. Manually detaching V tensor.")
                # If V was created but error occurred on requires_grad, manually detach it
                if hasattr(arb_q_model, 'V') and arb_q_model.V is not None:
                    arb_q_model.V = arb_q_model.V.detach()
                    logging.info("Successfully detached V tensor.")
            else:
                # Re-raise if it's a different error
                raise
        
        # Calculate covariance matrix if needed
        if hasattr(arb_q_model, 'compute_covariance_matrix'):
            try:
                arb_q_model.compute_covariance_matrix()
                logging.info("Covariance matrix calculated successfully")
            except Exception as e:
                logging.warning(f"Error calculating covariance matrix: {e}")
        
        # Verify phonon tensors exist and contain non-zero values
        if hasattr(arb_q_model, 'V') and hasattr(arb_q_model, 'Winv'):
            v_nonzero = torch.count_nonzero(torch.abs(arb_q_model.V)).item() if arb_q_model.V is not None else 0
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
        
        # Save results
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
        
        # Check if result files exist
        np_file = "np_diffuse_intensity.npy"
        torch_file = "torch_diffuse_intensity.npy"
        arb_q_file = "arb_q_diffuse_intensity.npy"
        
        files_exist = all(os.path.exists(f) for f in [np_file, torch_file, arb_q_file])
        if not files_exist:
            logging.warning("Not all result files exist. Please run all three methods first.")
            return
        
        # Load the data
        Id_np = np.load(np_file)
        Id_torch = np.load(torch_file)
        Id_arb_q = np.load(arb_q_file)
        
        # Get the map shape
        map_shape = Id_np.shape
        logging.info(f"Visualization: Map shape = {map_shape}")
        
        # Check if Id_arb_q needs reshaping
        if Id_arb_q.ndim == 1:
            logging.info(f"Reshaping arbitrary q-vector results from 1D to 3D ({Id_arb_q.shape} -> {map_shape})")
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
