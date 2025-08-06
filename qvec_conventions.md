# OnePhonon Implementation Conventions and Data Flow

This document establishes clear conventions for the OnePhonon implementation, focusing on arbitrary q-vector support and grid-based calculations.

## 1. Key Data Structures and Their Relationships

### 1.1. Core Data Structures

- **q_grid/q_vectors**: Reciprocal space vectors in Å⁻¹ (shape: [n_points, 3])
- **hkl_grid**: Miller indices corresponding to q-vectors (shape: [n_points, 3])
- **kvec**: Scaled q-vectors: k = q/(2π) (shape depends on mode)
- **kvec_norm**: Norm of k-vectors (shape depends on mode)
- **map_shape**: Shape of the 3D grid in grid-based mode

### 1.2. Mode Flag

- **use_arbitrary_q**: Boolean flag indicating whether arbitrary q-vector mode is active
  - Set to `True` when q_vectors parameter is provided in constructor
  - Set to `False` when using grid-based sampling (default)
  - Should be set in constructor and checked in all methods that handle both modes

## 2. Grid Parameter Interpretation

### 2.1. Sampling Parameters

Each sampling parameter (hsampling, ksampling, lsampling) is a tuple of (min, max, oversampling):

```python
# Example: Sample h from -2 to 2 with oversampling of 3
hsampling = (-2, 2, 3)
```

### 2.2. Grid Dimensions

The grid dimensions can be computed in two ways:

1. **Direct oversampling values**:
   ```python
   h_dim = int(hsampling[2])  # For hsampling=(-2, 2, 3), this gives h_dim=3
   ```

2. **Full dense grid from generate_grid**:
   ```python
   hsteps = int(hsampling[2] * (hsampling[1] - hsampling[0]) + 1)
   # For hsampling=(-2, 2, 3), this gives hsteps=13
   ```

### 2.3. Standardized Grid Convention

For **consistency with the NumPy implementation and tests**, we will use the following convention:

- In grid-based mode, `map_shape` should be set to the output of `generate_grid`
- All internal calculations should respect this full grid shape
- Methods like `_flat_to_3d_indices` should use these dimensions

**Example**:
```python
# For sampling (-2, 2, 3):
# Original NumPy model produces a grid with 13×13×13 = 2197 points
# Our implementation should match this exactly
```

## 3. Methods That Use Grid/K/Q Data

| Method | Grid Mode | Arbitrary Q Mode | Data Used |
|--------|-----------|------------------|-----------|
| `_setup` | Initializes q_grid, hkl_grid using generate_grid | Uses provided q_vectors directly | map_shape, q_grid, hkl_grid |
| `_build_kvec_Brillouin` | Computes kvec from hkl using A_inv | Sets kvec = q_grid/(2π) | q_grid, hkl_grid, kvec, kvec_norm |
| `_at_kvec_from_miller_points` | Returns indices in grid using Miller indices | Returns nearest q-vector index | q_grid, hkl_grid, map_shape |
| `compute_gnm_phonons` | Computes phonon modes for grid points | Computes modes for arbitrary points | kvec, V, Winv |
| `compute_covariance_matrix` | Computes covariance using grid structure | Computes without grid structure | kvec, covar, ADP |
| `apply_disorder` | Iterates through grid points | Processes arbitrary points directly | q_grid, V, Winv, res_mask |
| `to_batched_shape` | Reshapes [h,k,l,...] to [h*k*l,...] | Identity operation | map_shape |
| `to_original_shape` | Reshapes [h*k*l,...] to [h,k,l,...] | Identity operation | map_shape |

## 4. Mapping Between Grid and Arbitrary Q-vectors

### 4.1. When Is Mapping Needed?

1. **Testing**: Comparing outputs between grid-based and arbitrary modes
2. **Visualization**: Organizing arbitrary q-vector results in a grid for visualization
3. **Compatibility**: Interfacing with methods that expect grid-structured data

### 4.2. How Should Mapping Be Implemented?

For arbitrary q-vector mode, we implement two key operations:

1. **Miller-to-Index Mapping**:
   ```python
   def _at_kvec_from_miller_points(self, indices_or_batch):
       if self.use_arbitrary_q:
           if isinstance(indices_or_batch, (tuple, list)) and len(indices_or_batch) == 3:
               # Convert Miller indices to q-vector and find nearest point
               hkl = torch.tensor([indices_or_batch], device=self.device)
               A_inv_tensor = torch.tensor(self.model.A_inv, device=self.device)
               target_q = 2 * torch.pi * torch.matmul(A_inv_tensor.T, hkl.T).T
               
               # Find nearest q-vector by distance
               distances = torch.norm(self.q_grid - target_q, dim=1)
               nearest_idx = torch.argmin(distances)
               return nearest_idx
           else:
               # Direct index, return as-is
               return indices_or_batch
   ```

2. **Shape Conversion**:
   ```python
   def to_original_shape(self, tensor):
       if self.use_arbitrary_q:
           # In arbitrary mode, no reshaping needed
           return tensor
       else:
           # In grid mode, reshape to 3D grid
           h_dim, k_dim, l_dim = self.map_shape
           return tensor.reshape(h_dim, k_dim, l_dim, *tensor.shape[1:])
   ```

### 4.3. Representation: Object Attributes vs Function-Based

The mapping should be represented as **object attributes** with functional lookup methods:

- **Storage**: q_grid, hkl_grid, map_shape attributes (initialized in `_setup`)
- **Lookup**: `_at_kvec_from_miller_points` method

## 5. Implementation Priorities and Best Practices

### 5.1. Consistency Priority Rules

1. Set `use_arbitrary_q` in constructor for ALL instances (even when False)
2. Use `self.map_shape` consistently throughout the code
3. Always check `if getattr(self, 'use_arbitrary_q', False):` when branching
4. Ensure shape conversion methods never return None
5. Preserve gradient flow in all operations involving q_vectors

### 5.2. Method Stubs and Examples

```python
# Constructor
def __init__(self, pdb_path, hsampling=None, ksampling=None, lsampling=None, q_vectors=None, ...):
    # Set mode flag for ALL instances
    self.use_arbitrary_q = False  # Default to grid mode
    
    if q_vectors is not None:
        # Validate q_vectors
        if not isinstance(q_vectors, torch.Tensor):
            raise ValueError("q_vectors must be a PyTorch tensor")
        if q_vectors.dim() != 2 or q_vectors.shape[1] != 3:
            raise ValueError(f"q_vectors must have shape [n_points, 3], got {q_vectors.shape}")
            
        # Switch to arbitrary mode
        self.use_arbitrary_q = True
        self.q_vectors = q_vectors.to(device=self.device)
        self.q_vectors.requires_grad_(True)
    elif hsampling is None or ksampling is None or lsampling is None:
        raise ValueError("Either q_vectors or all sampling parameters must be provided")
        
    # Continue with setup...
```

```python
# apply_disorder method
def apply_disorder(self, rank=-1, outdir=None, use_data_adp=False):
    # Initialize intensity tensor
    Id = torch.zeros(self.q_grid.shape[0], dtype=torch.float32, device=self.device)
    
    if getattr(self, 'use_arbitrary_q', False):
        # Arbitrary q-vector mode
        valid_indices = torch.where(self.res_mask)[0]
        if valid_indices.numel() > 0:
            # Perform calculation for arbitrary points...
    else:
        # Grid-based mode
        # Use map_shape for dimensions
        h_dim, k_dim, l_dim = self.map_shape
        # Process grid points...
    
    # Apply resolution mask
    Id_masked = Id.clone()
    Id_masked[~self.res_mask] = float('nan')
    
    return Id_masked
```

## 6. Component Relationships and Data Dependencies

```
┌─────────────────┐     ┌─────────────────┐
│   Constructor   │     │ apply_disorder  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│     _setup      │     │ structure_factors│
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│_build_kvec_Brill│     │   compute_K     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│compute_gnm_phonon     │compute_covariance│
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│to_batched_shape │
└─────────────────┘
```

### Data Flow Path

1. **Initialization**: Constructor → _setup → _build_kvec_Brillouin
2. **Phonon Computation**: compute_gnm_phonons → compute_covariance_matrix
3. **Intensity Calculation**: apply_disorder → structure_factors → compute_K

### Critical Gradient Paths

For arbitrary q-vector mode, the gradient flow must be preserved in:
1. q_vectors → q_grid → kvec → compute_gnm_phonons → V, Winv
2. V, Winv → apply_disorder → Id (intensity)

## 7. Required Fixes for Current Implementation

1. **Gradient Flow Fix**: Ensure all operations on q_vectors preserve gradients
2. **Consistent Grid Dimensions**: Use full generate_grid dimensions for grid mode
3. **Shape Conversion Robustness**: Ensure to_original_shape never returns None
4. **Attribute Safety**: Use getattr for use_arbitrary_q checks
5. **Resolution Mask Consistency**: Apply the same resolution criteria in both modes
6. **Broadcasting Fix**: Ensure tensor shapes match in apply_disorder calculations

## 8. Testing Strategy

1. **Unit Tests**: Test each method in isolation
2. **Integration Tests**: Test end-to-end calculations
3. **Gradient Tests**: Verify gradient flow for optimization
4. **Equivalence Tests**: Ensure grid-based and arbitrary modes produce equivalent results when using the same q-vectors
5. **Performance Tests**: Benchmark performance between modes
