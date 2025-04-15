https://claude.ai/chat/08448094-a163-4569-9501-4d5ffb1027a0 

# Comprehensive Guide to Fixing Arbitrary Q-Vector Mode in OnePhononTorch

## Overview

This guide provides an in-depth understanding of how arbitrary q-vector mode works in OnePhononTorch, why it produces zero intensities by default, and how to fix it while preserving differentiability. It's based on a detailed analysis of the codebase and a careful consideration of the underlying physics.

## Key Issues and Insights

### 1. Zero Intensities Issue

Arbitrary q-vector mode intentionally skips phonon calculations during initialization:

```python
# In _setup_phonons method
if getattr(self, 'use_arbitrary_q', False):
    logging.debug("[_setup_phonons] Skipping full phonon computation in arbitrary q-vector mode")
else:
    self.compute_gnm_phonons()
    self.compute_covariance_matrix()
```

This skipping is the root cause of zero intensities when using arbitrary q-vector mode.

### 2. Gradient Tracking Issue

When manually calling `compute_gnm_phonons()` to fix the zero intensities, an error occurs:

```
RuntimeError: you can only change requires_grad flags of leaf variables. If you want to use a 
computed variable in a subgraph that doesn't require differentiation use var_no_grad = var.detach().
```

This happens in the last few lines of `compute_gnm_phonons()`:

```python
# V should NOT require grad as it uses detached eigenvectors
self.V.requires_grad_(False)  # <-- This line causes the error
# Winv SHOULD require grad if eigenvalues_all did
self.Winv.requires_grad_(eigenvalues_all.requires_grad)
```

### 3. BZ-Averaged ADPs vs. Covariance Matrix

A critical insight is that ADPs (Atomic Displacement Parameters) should be calculated by integrating over the entire Brillouin zone, independent of the arbitrary q-vectors:

```python
# In _setup_phonons
if self.model_type == 'gnm':
    try:
        self.bz_averaged_adp = self._compute_bz_averaged_adp()
        # ...
    except Exception as e:
        logging.error(f"[_setup_phonons] Failed to compute BZ-averaged ADP: {e}")
        self.bz_averaged_adp = None
```

This BZ-averaging is the physically correct approach for calculating ADPs, which don't depend on specific q-vectors.

### 4. Differentiability Advantage

The independent calculation of phonon modes for each arbitrary q-vector preserves differentiability with respect to q, which is essential for gradient-based optimization. Any BZ mapping approach would break this differentiability.

## Understanding the Physical Model

### The Correct Physical Approach:

1. **For ADPs/Debye-Waller Factors**:
   - Calculate via BZ integration using a uniform grid (defined by hsampling, ksampling, lsampling)
   - Mathematical form: `ADP_i = (1/N) ∑_q [ eigenvectors_q * (1/frequency_q²) * eigenvectors_q† ]_ii`
   - This summation is over all BZ k-vectors, totally independent of the arbitrary q-vectors

2. **For Phonon Modes (V, Winv)**:
   - Calculate independently for each arbitrary q-vector
   - These determine the intensity distribution at each specific q-vector
   - Independent calculation preserves differentiability

3. **For Intensity Calculation**:
   - Use BZ-averaged ADPs for Debye-Waller factors in structure factor calculation
   - Use q-specific phonon modes for intensity distribution

This is what the code already attempts to do, but with the missing phonon calculation and gradient tracking issue.

## Implementation Solutions

### 1. Immediate Workaround

```python
def run_arbitrary_q_mode(pdb_path, q_vectors, sampling_params, **kwargs):
    # Initialize model
    model = OnePhononTorch(
        pdb_path=pdb_path, 
        q_vectors=q_vectors,
        hsampling=sampling_params[0],  # Required for BZ-averaged ADP
        ksampling=sampling_params[1], 
        lsampling=sampling_params[2],
        **kwargs
    )
    
    # Explicitly calculate phonon modes with gradient handling
    try:
        model.compute_gnm_phonons()
    except RuntimeError as e:
        if "requires_grad flags of leaf variables" in str(e):
            # Safe detach that preserves the computation graph
            if hasattr(model, 'V') and model.V is not None:
                model.V = model.V.detach()
                logging.info("Successfully detached V tensor for gradient flow.")
    
    # Do NOT call compute_covariance_matrix() - it would overwrite
    # the correctly BZ-averaged ADPs with values from arbitrary q-vectors
    
    # Verify phonon tensors contain non-zero values
    if hasattr(model, 'V') and hasattr(model, 'Winv'):
        v_nonzero = torch.count_nonzero(torch.abs(model.V)).item()
        winv_nonzero = torch.count_nonzero(~torch.isnan(model.Winv)).item()
        logging.info(f"Phonon tensors - V: {v_nonzero} non-zeros, Winv: {winv_nonzero} non-NaNs")
    
    # Calculate intensity
    intensity = model.apply_disorder()
    
    return model, intensity
```

### 2. Medium-Term Solution: Subclass Implementation

```python
class DifferentiableOnePhonon(OnePhononTorch):
    """OnePhonon implementation with fixed arbitrary q-vector mode."""
    
    def __init__(self, pdb_path, q_vectors=None, **kwargs):
        # Call parent constructor
        super().__init__(pdb_path, q_vectors=q_vectors, **kwargs)
        
        # If in arbitrary q-vector mode, automatically calculate phonons
        if getattr(self, 'use_arbitrary_q', False):
            self._calculate_phonons_safely()
    
    def _calculate_phonons_safely(self):
        """Safely calculate phonon modes preserving differentiability."""
        try:
            # Attempt normal phonon calculation
            self.compute_gnm_phonons()
        except RuntimeError as e:
            if "requires_grad flags of leaf variables" in str(e):
                # If V was created but error occurred on requires_grad, manually detach it
                if hasattr(self, 'V') and self.V is not None:
                    self.V = self.V.detach()
                    logging.info("Successfully detached V tensor for gradient flow.")
            else:
                # Re-raise if it's a different error
                raise
```

### 3. Long-Term Solution: Core Implementation Changes

For a permanent fix, these changes to the original code are needed:

1. **Fix `compute_gnm_phonons`** to avoid the requires_grad issue:

```python
# Replace the problematic lines at the end:
self.V.requires_grad_(False)
self.Winv.requires_grad_(eigenvalues_all.requires_grad)

# With:
# Create properly detached V tensor directly
self.V = torch.matmul(Linv_H_batch, v_all_detached).detach()
# Maintain gradients for Winv as needed
self.Winv = winv_all.to(dtype=self.complex_dtype)
if not eigenvalues_all.requires_grad:
    self.Winv = self.Winv.detach()
```

2. **Update `_setup_phonons`** to calculate phonons in arbitrary q-vector mode:

```python
# Replace:
if getattr(self, 'use_arbitrary_q', False):
    logging.debug("[_setup_phonons] Skipping full phonon computation in arbitrary q-vector mode")
else:
    self.compute_gnm_phonons()
    self.compute_covariance_matrix()

# With:
# Always calculate phonons for both modes
self.compute_gnm_phonons()
# Only calculate covariance matrix in grid mode
if not getattr(self, 'use_arbitrary_q', False):
    self.compute_covariance_matrix()
```

## Next Steps and Recommendations

1. **Immediate Usage**:
   - Implement the workaround function for arbitrary q-vector mode
   - Ensure sampling parameters are passed for proper ADP calculation
   - Do NOT call compute_covariance_matrix() in arbitrary q-vector mode

2. **Testing and Validation**:
   - Compare intensity patterns with grid mode results
   - Verify the preservation of differentiability by checking gradients
   - Test with various q-vector patterns

3. **Performance Optimization**:
   - Implement batched computation of phonon modes
   - Add caching for similar q-vectors without breaking gradient flow
   - Profile memory usage for large systems

4. **Documentation Updates**:
   - Document the physical model in arbitrary q-vector mode
   - Clarify the ADPs calculation via BZ integration
   - Provide examples of gradient-based optimization

5. **Scientific Applications**:
   - Explore machine learning applications with gradient-based optimization
   - Develop custom q-vector sampling strategies for specific experimental setups
   - Investigate the scientific implications of differentiable diffuse scattering models

By implementing these fixes and following these recommendations, arbitrary q-vector mode can be a powerful tool for advanced applications in thermal diffuse scattering analysis with differentiability advantages.

---

## Grid vs. Arbitrary Q-Vector Modes in `OnePhononTorch`

### 1. Overview

The `OnePhononTorch` class calculates diffuse scattering intensity and supports two primary modes of operation regarding the scattering vectors (**q**) at which the intensity is computed:

*   **Grid Mode:** This is the default and traditional mode. The user specifies sampling parameters (`hsampling`, `ksampling`, `lsampling`) defining a regular grid in reciprocal space (hkl space). The model calculates intensity at all points on this grid. This mode leverages the periodicity of the crystal lattice to optimize phonon calculations.
*   **Arbitrary Q-Vector Mode:** The user directly provides a list (as a `torch.Tensor`) of specific **q**-vectors of interest. The model calculates intensity *only* at these provided points. This mode offers flexibility for targeted calculations (e.g., comparing with specific experimental data points) but currently performs less optimization.

### 2. Notation Clarification (q vs. k, BZ vs. Full)

Understanding the distinction between **q** and **k** is crucial:

*   **q (Scattering Vector):**
    *   Represents the momentum transfer vector in a scattering experiment (**q** = **k**<sub>final</sub> - **k**<sub>initial</sub>).
    *   Directly related to Miller indices (hkl) and the reciprocal lattice: **q** = 2π (**A**<sup>-1</sup>)<sup>T</sup> **hkl**.
    *   Units: Typically Å⁻¹.
    *   **Used for:** Calculating structure factors `F(q)` (including atomic form factors and Debye-Waller factors, which depend on |q|).
    *   **In the code:** `self.q_grid` (stores the q-vectors for which intensity is calculated), `q_vectors` (user input in arbitrary mode).

*   **k (Phonon Wavevector):**
    *   Represents the wavevector of a lattice vibration mode (phonon).
    *   Due to crystal periodicity, all *unique* phonon modes are described by **k**-vectors within the **First Brillouin Zone (BZ)**.
    *   Any **k** outside the BZ is equivalent to a **k'** inside the BZ plus a reciprocal lattice vector **G** (i.e., **k** = **k'** + **G**). Both **k** and **k'** describe the *same physical vibration pattern*.
    *   Units: Typically Å⁻¹.
    *   **Used for:** Calculating phonon properties: Dynamical Matrix `K(k)`, mass-weighted Dynamical Matrix `D(k)`, eigenvectors `V(k)`, and eigenvalues `ω²(k)` (related to `Winv(k)`).
    *   **In the code:** `self.kvec` (stores the k-vectors used for phonon calculations), `kvec_bz_local` (generated inside `compute_covariance_matrix`).

*   **BZ k-vectors vs. Full Grid k-vectors vs. Arbitrary k-vectors:**
    *   **BZ k-vectors:** The minimal, unique set of k-vectors within the first BZ needed to describe all phonon modes. Their number (`n_bz_points`) is determined by the *sampling rates* (`hsampling[2]`, `ksampling[2]`, `lsampling[2]`). Grid mode primarily calculates phonons for these.
    *   **Full Grid q/k-vectors:** The complete set of points defined by the grid limits and sampling (`hsampling[0]`, `hsampling[1]`, etc.). Their number (`n_full_points`) is `product(map_shape)`. Each full grid q-vector maps to one of the BZ k-vectors.
    *   **Arbitrary q/k-vectors:** The user-provided list. No inherent grid structure or assumed relationship between points. `k` is derived directly as `q / 2π`. Their number (`n_arb_points`) is the length of the input list.

### 3. Mathematical Description

*   **Core Intensity Calculation (Same for Both Modes):**
    The fundamental equation for one-phonon diffuse scattering intensity `I(q)` at scattering vector `q` is approximated by summing over phonon modes `m`:
    `I(q) ≈ Sum_m [ | F(q) · V_m(k) |² * Winv_m(k) ]`
    where:
    *   `F(q)` is the structure factor vector at `q`.
    *   `k` is the phonon wavevector corresponding to `q`.
    *   `V_m(k)` is the eigenvector (polarization vector) for mode `m` at wavevector `k`.
    *   `Winv_m(k)` is related to the inverse squared frequency (1/ω²) for mode `m` at `k`.

*   **Structure Factor `F(q)` (Same for Both Modes):**
    Depends only on `q` and atomic properties (positions `r`, form factors `f`, ADPs `B`):
    `F(q) = Sum_atoms [ f_atom(q) * exp(i * q · r_atom) * DWF(q, B) ]`
    *Note:* The ADP `B` used in the Debye-Waller Factor `DWF` depends on the `use_data_adp` flag. If `False`, `B` comes from `self.ADP`.

*   **Phonon Modes `V(k), Winv(k)` (Same Calculation, Different Inputs):**
    Derived from the dynamical matrix `D(k)`, which comes from the Hessian `H`:
    1.  `K(k) = Sum_cells [ H(cell) * exp(i * k · R_cell) ]`
    2.  `D(k) = Linv @ K(k) @ Linv.H`
    3.  `eigenvalues(ω²), eigenvectors(v) = eigendecomposition(D(k))`
    4.  `V(k) = Linv.H @ v`
    5.  `Winv(k) = 1 / ω²` (with thresholding)
    The *calculation steps* are the same, but the set of `k` vectors for which this is performed differs between modes.

*   **ADP Calculation `B ~ <u^2>` (Same for Both Modes):**
    Requires averaging `Kinv(k)` over the Brillouin Zone:
    `<u^2>_atom ~ Diagonal [ Average_k_BZ [ Kinv(k_BZ) ] ]` (followed by projection, sum, scaling)
    This calculation *must* use a representative sample of the BZ (`k_BZ`), independent of the `q` vectors requested for intensity.

*   **Key Mathematical Difference:**
    *   **Grid Mode:** For a given `q` on the full grid, the corresponding `k` used in the intensity formula is `k = k_BZ(q)`, where `k_BZ` is the equivalent wavevector *within the first BZ*. Phonon modes `V(k_BZ)` and `Winv(k_BZ)` are looked up.
    *   **Arbitrary-Q Mode:** For a given input `q`, the corresponding `k` used is `k = q / 2π` (direct conversion). Phonon modes `V(k)` and `Winv(k)` are calculated *specifically for this k*.

### 4. Computational Flow & Key Differences

| Step                          | Grid Mode                                                                 | Arbitrary Q-Vector Mode                                                        | Key Difference                                                                 |
| :---------------------------- | :------------------------------------------------------------------------ | :----------------------------------------------------------------------------- | :----------------------------------------------------------------------------- |
| **`__init__`**                | Sets `use_arbitrary_q = False`.                                           | Sets `use_arbitrary_q = True`.                                                 | Mode flag set.                                                                 |
| **`_setup`**                  | Generates `hkl_grid`, `q_grid`, `map_shape` based on sampling params.     | Uses input `q_vectors` as `q_grid`. Calculates `hkl_grid`. Sets dummy `map_shape`. | Source and structure of `q_grid`, `map_shape`.                               |
| **`_build_kvec_Brillouin`**   | Calculates `self.kvec` for unique BZ points (`n_bz_points`).              | Calculates `self.kvec` for *all* input `q_vectors` (`n_arb_points`).         | **Size and content of `self.kvec`**.                                         |
| **`compute_gnm_phonons`**     | Calculates `V`, `Winv` for `n_bz_points` BZ k-vectors.                    | Calculates `V`, `Winv` for `n_arb_points` arbitrary k-vectors.                 | **Size of `V`, `Winv` tensors and the k-vectors they correspond to.**        |
| **`compute_covariance_matrix`** | **(Corrected)** Generates BZ k-vectors locally, computes `Kinv` for BZ points, averages over BZ points to get `self.ADP`. | **(Corrected)** Generates BZ k-vectors locally, computes `Kinv` for BZ points, averages over BZ points to get `self.ADP`. | **Calculation is now identical and mode-independent.**                       |
| **`apply_disorder`**          |                                                                           |                                                                                |                                                                                |
| - ADP Selection             | Uses `self.ADP` (from BZ avg) if `use_data_adp=False`.                    | Uses `self.ADP` (from BZ avg) if `use_data_adp=False`.                    | **Selection logic is now identical.**                                        |
| - Structure Factor `F(q)`   | Calculates `F` for subset of `q_grid` points relevant to current BZ point. | Calculates `F` for all valid input `q_vectors`.                               | Input `q` points for `F` calculation.                                        |
| - Phonon Lookup             | Maps `q` to `bz_idx`. Uses `V[bz_idx]`, `Winv[bz_idx]`.                     | Uses direct index `i`. Uses `V[i]`, `Winv[i]`.                                 | **How `V` and `Winv` are accessed (using which index/k-vector).**            |
| - Intensity Summation       | Accumulates intensity into full grid `Id` using `index_add_`.             | Calculates intensity for each valid `q` and places it in final `Id` tensor.    | Output construction method.                                                    |

### 5. Project Conventions for Comparison/Visualization

*   **Data Saving (`run_debug.py`):** Save results as `.npz` files containing:
    *   `q_vectors`: The actual q-vectors used (shape `[N, 3]`).
    *   `intensity`: The corresponding *flat* intensity array (shape `[N]`).
    *   `map_shape`: The 3D grid shape `(H, K, L)` (even for arbitrary-q mode, save the shape derived from the grid run for potential reshaping).
*   **Data Loading/Visualization (`visualize_diffuse.py`):**
    *   Load the `.npz` file.
    *   Use the `map_shape` key to attempt reshaping the flat `intensity` array into a 3D array.
    *   If reshaping fails (e.g., size mismatch, `map_shape` missing) or data is inherently not grid-like, proceed with 1D data (e.g., only plot histogram).
    *   Use the `q_vectors` array for any q-dependent analysis or plotting.
*   **Equivalence Testing:**
    *   Run grid mode, save `q_grid`, `V_grid`, `Winv_grid`, `kvec_grid` (BZ k-vectors).
    *   Run arbitrary-q mode using saved `q_grid`.
    *   Map arbitrary-q `kvec` values back to BZ indices using `kvec_grid`.
    *   Inject `V_grid[map]` and `Winv_grid[map]` into the arbitrary-q model instance before calling `apply_disorder`.
    *   Compare the final intensity outputs.

### 6. Issues Solved

*   Removed erroneous `bz_averaged_adp` calculation and attribute.
*   Corrected ADP selection logic in `apply_disorder` to depend only on `use_data_adp`.
*   Made `compute_covariance_matrix` mode-agnostic, ensuring it always averages over the BZ grid defined by sampling parameters to calculate `self.ADP`.

### 7. Remaining Issues

*   **Large Scale Difference:** Equivalence tests comparing the final intensity output of grid mode vs. arbitrary-q mode (even when using the *same* input q-vectors and `use_data_adp=True`) show a **large difference in overall intensity scale**. This indicates the core intensity calculation steps within `apply_disorder` are yielding different results despite nominally having the same inputs at corresponding q-points.

### 8. Hypotheses for Underlying Causes

1.  **Incorrect Phonon Data Usage in `apply_disorder`:** The primary suspect. Even if `V`/`Winv` are calculated *differently* in the two modes (`n_bz` vs `n_arb` points), the equivalence test *should* pass if `apply_disorder` correctly retrieves and uses the corresponding phonon data for each `q`.
    *   **Grid Mode:** Is the mapping from `q_grid` index to `bz_idx` correct? Is `V[bz_idx]` and `Winv[bz_idx]` being accessed properly within the BZ loops?
    *   **Arbitrary-Q Mode:** Is `V[i]` and `Winv[i]` being accessed correctly for the i-th q-vector?
2.  **Subtle Differences in `compute_gnm_phonons` Output:** Although mathematically equivalent points should yield the same physics, are there numerical scaling differences between `V_grid[bz_idx]` and `Winv_grid[bz_idx]` compared to `V_q[i]` and `Winv_q[i]` (where `q[i]` maps to `bz_idx`) large enough to cause the scale difference? This seems less likely for *large* differences but possible.
3.  **Structure Factor (`F`) Calculation Discrepancy:** Is `structure_factors_batch` somehow producing differently scaled results when called with the full batch of arbitrary q-vectors versus the smaller batches within the grid mode's BZ loop, even with identical ADP? (Unlikely unless there's a batch-size dependent bug).
4.  **Precision Issues:** Are float64/complex128 types being consistently used in *all* intermediate steps of the intensity calculation (`F`, `V`, `Winv`, matmuls, sum) in both modes? A mismatch could cause scaling issues.

### 9. Next Steps for Resolution

1.  **Implement Revised Equivalence Test:** Create the test described in Section 5 (Project Conventions) that *injects* the `V_grid` and `Winv_grid` data (mapped correctly) into the `model_q` instance before calling `apply_disorder`.
    *   **If this test passes:** The core summation logic in `apply_disorder` is likely correct for both modes. The problem lies in the fact that `compute_gnm_phonons` produces effectively different `V`/`Winv` tensors used by the two modes (due to calculating for `n_bz` vs `n_arb` k-vectors). The "fix" is then to accept this difference or implement one of the efficiency strategies (like BZ mapping/grouping) for the arbitrary-q mode's phonon calculation if grid-equivalent results are desired.
    *   **If this test fails:** The problem lies *within* the `apply_disorder` method itself – likely in how `F`, `V`, or `Winv` are indexed, accessed, or combined differently between the `if use_arbitrary_q:` and `else:` blocks.
2.  **Detailed Intermediate Comparison (if step 1 fails):**
    *   Choose a specific `q_vector` index `i` present in both runs.
    *   Determine its corresponding BZ index `bz_i`.
    *   Add print statements inside `apply_disorder` for both modes to log:
        *   The `F` vector calculated for `q_grid[i]`.
        *   The `V` matrix used (`V_grid[bz_i]` vs `V_q[i]`).
        *   The `Winv` vector used (`Winv_grid[bz_i]` vs `Winv_q[i]`).
        *   The intermediate `FV = F @ V` result.
        *   The final intensity contribution `Sum(|FV|^2 * Winv)`.
    *   Compare these intermediate values step-by-step to pinpoint where the large scale difference originates.
3.  **Verify Precision:** Double-check `dtype` usage (`real_dtype`, `complex_dtype`) throughout `apply_disorder` and its dependencies (`structure_factors_batch`, complex ops) to ensure consistency.

