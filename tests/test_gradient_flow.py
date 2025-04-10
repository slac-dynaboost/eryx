import unittest
import torch
import numpy as np
from eryx.models_torch import OnePhonon

class TestGradientFlow(unittest.TestCase):
    """Test suite for gradient flow in OnePhonon with arbitrary q-vectors."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device('cpu')
        self.pdb_path = "tests/pdbs/5zck_p1.pdb"
    
#    def test_gradient_to_q_vectors(self):
#        """Test that gradients flow back to q_vectors."""
#        # Create model with arbitrary q-vectors that require gradients
#        q_vectors = torch.tensor(
#            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
#            device=self.device,
#            requires_grad=True
#        )
#        
#        # Create optimizer to track gradients
#        optimizer = torch.optim.Adam([q_vectors], lr=0.01)
#        
#        # Define hooks to track gradient flow
#        def hook_fn(name):
#            def hook(grad):
#                print(f"Gradient for {name}: {grad.norm().item() if grad is not None else 'None'}")
#                return grad
#            return hook
#        
#        # Register hooks
#        q_vectors.register_hook(hook_fn("q_vectors"))
#        
#        # Initialize model
#        model = OnePhonon(
#            self.pdb_path,
#            q_vectors=q_vectors,
#            device=self.device
#        )
#        
#        # Register hook for kvec
#        model.kvec.register_hook(hook_fn("kvec"))
#        
#        # Verify kvec has correct relationship with q_vectors
#        # k = q/(2π)
#        expected_kvec = q_vectors / (2.0 * torch.pi)
#        self.assertTrue(torch.allclose(model.kvec, expected_kvec, rtol=1e-5))
#        
#        # Compute phonons - key step where gradients need to flow
#        model.compute_gnm_phonons()
#        
#        # Compute covariance matrix
#        model.compute_covariance_matrix()
#        
#        # Zero gradients
#        optimizer.zero_grad()
#        
#        # Create a loss from ADP
#        # Filter out NaN values for stable loss computation
#        valid_adp = model.ADP[~torch.isnan(model.ADP)]
#        if valid_adp.numel() > 0:
#            loss = torch.sum(valid_adp)
#        else:
#            # Fallback if all values are NaN - use a direct connection to kvec
#            loss = torch.sum(model.kvec)
#        
#        print(f"Loss value: {loss.item()}")
#        
#        # Backpropagate
#        loss.backward()
#        
#        # Verify gradients exist and are non-zero
#        self.assertIsNotNone(q_vectors.grad)
#        self.assertGreater(torch.norm(q_vectors.grad), 0.0)
#        
#        # Print gradient statistics for debugging
#        print(f"q_vectors.grad norm: {torch.norm(q_vectors.grad)}")
#        print(f"q_vectors.grad: {q_vectors.grad}")
#        
#        # Run optimizer step to verify it changes q_vectors
#        q_vectors_before = q_vectors.clone()
#        optimizer.step()
#        
#        # Verify q_vectors changed after optimization
#        self.assertFalse(torch.allclose(q_vectors, q_vectors_before))

    def test_gradient_to_q_vectors(self):
        """Test that gradients flow back to q_vectors."""
        # Create model with arbitrary q-vectors that require gradients
        q_vectors = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            device=self.device,
            requires_grad=True
        )
        
        # Register gradient hooks for debugging
        def hook_fn(name):
            def hook(grad):
                print(f"Gradient for {name}: {grad.norm().item() if grad is not None else 'None'}")
                return grad
            return hook
        
        q_vectors.register_hook(hook_fn("q_vectors"))
        
        # Create optimizer to track gradients
        optimizer = torch.optim.Adam([q_vectors], lr=0.01)
        
        # Initialize model
        model = OnePhonon(
            self.pdb_path,
            q_vectors=q_vectors,
            device=self.device
        )
        
        model.kvec.register_hook(hook_fn("kvec"))
        
        # Compute phonons
        model.compute_gnm_phonons()
        
        # Compute covariance matrix
        model.compute_covariance_matrix()
        
        # Ensure gradients are zero before loss computation
        optimizer.zero_grad()
        
        # Create a loss by explicitly using all elements of ADP to maintain gradient connections
        # Avoid any operations that might break gradient flow
        adp_clone = model.ADP.clone()  # Clone to avoid potential in-place modifications
        loss = torch.sum(adp_clone)    # Simple sum to preserve gradient flow
        
        print(f"Loss value: {loss.item()}")
        print(f"ADP requires_grad: {adp_clone.requires_grad}")
        print(f"Loss requires_grad: {loss.requires_grad}")
        
        # Backpropagate with retain_graph to ensure all gradients are computed
        loss.backward(retain_graph=True)
        
        # Verify gradients exist and are non-zero
        print(f"After backward - q_vectors.grad: {q_vectors.grad}")
        print(f"After backward - q_vectors.grad norm: {torch.norm(q_vectors.grad).item()}")
        
        self.assertIsNotNone(q_vectors.grad)
        self.assertGreater(torch.norm(q_vectors.grad), 0.0)
    
    def test_end_to_end_gradient_flow(self):
        """Test gradient flow through the entire pipeline including apply_disorder."""
        # Create a small test case for faster test execution
        q_vectors = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            device=self.device,
            requires_grad=True
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam([q_vectors], lr=0.01)
        
        # Define hooks to track gradient flow
        def hook_fn(name):
            def hook(grad):
                print(f"Gradient for {name}: {grad.norm().item() if grad is not None else 'None'}")
                return grad
            return hook
        
        # Register hooks
        q_vectors.register_hook(hook_fn("q_vectors"))
        
        # Initialize model
        model = OnePhonon(
            self.pdb_path,
            q_vectors=q_vectors,
            device=self.device
        )
        
        # Register hook for kvec
        model.kvec.register_hook(hook_fn("kvec"))
        
        # Compute phonons
        model.compute_gnm_phonons()
        
        # Compute covariance matrix
        model.compute_covariance_matrix()
        
        # Apply disorder to get diffuse intensity
        intensity = model.apply_disorder(use_data_adp=True)
        
        # Replace NaN values with zeros for loss computation
        intensity_no_nan = torch.nan_to_num(intensity, 0.0)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Create a simple loss function
        loss = torch.sum(intensity_no_nan)
        print(f"Loss value: {loss.item()}")
        
        # Track key tensors before backward
        key_tensors = {
            'q_vectors': q_vectors,
            'kvec': model.kvec,
            'V': model.V,
            'Winv': model.Winv,
            'ADP': model.ADP,
        }
        
        # Backpropagate
        loss.backward()
        
        # Verify gradients for all key tensors
        self.assertIsNotNone(q_vectors.grad)
        self.assertGreater(torch.norm(q_vectors.grad), 0.0)
        
        # Track gradient flow through model
        if hasattr(model, '_track_gradient_flow'):
            grad_stats = model._track_gradient_flow(key_tensors)
            print("Gradient statistics:")
            for name, stats in grad_stats.items():
                print(f"  {name}: {stats}")
        
        # Run optimizer step and verify q_vectors changed
        q_vectors_before = q_vectors.clone()
        optimizer.step()
        self.assertFalse(torch.allclose(q_vectors, q_vectors_before))
    
    def test_gradient_through_kvec(self):
        """Test that gradients flow through kvec calculations."""
        # Create model with arbitrary q-vectors
        q_vectors = torch.tensor(
            [[0.1, 0.2, 0.3]],
            device=self.device,
            requires_grad=True
        )
        
        model = OnePhonon(
            self.pdb_path,
            q_vectors=q_vectors,
            device=self.device
        )
        
        # Create a simple loss directly from kvec
        loss = torch.sum(model.kvec)
        
        # Backpropagate
        loss.backward()
        
        # Verify gradient on q_vectors
        self.assertIsNotNone(q_vectors.grad)
        
        # The gradient should be 1/(2π) for each component
        expected_grad = torch.ones_like(q_vectors) / (2.0 * torch.pi)
        self.assertTrue(torch.allclose(q_vectors.grad, expected_grad, rtol=1e-5))

if __name__ == '__main__':
    unittest.main()
