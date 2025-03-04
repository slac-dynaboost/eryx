"""
Unit tests for PyTorch utility functions.

This module contains tests for the PyTorch utility classes in eryx.torch_utils,
including complex number operations, eigendecomposition, and gradient utilities.
"""

import unittest
import torch
import numpy as np
from eryx.torch_utils import ComplexTensorOps, EigenOps, GradientUtils

class TestComplexTensorOps(unittest.TestCase):
    """Test cases for ComplexTensorOps class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set default device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Set default dtype
        torch.set_default_dtype(torch.float64)  # Use double precision for better accuracy in tests
        
    def test_complex_exp(self):
        """Test complex exponential function."""
        # Test zero phase
        phase = torch.tensor(0.0, device=self.device)
        real, imag = ComplexTensorOps.complex_exp(phase)
        self.assertAlmostEqual(real.item(), 1.0, places=6)
        self.assertAlmostEqual(imag.item(), 0.0, places=6)
        
        # Test π/2 phase
        phase = torch.tensor(np.pi/2, device=self.device)
        real, imag = ComplexTensorOps.complex_exp(phase)
        self.assertAlmostEqual(real.item(), 0.0, places=6)
        self.assertAlmostEqual(imag.item(), 1.0, places=6)
        
        # Test π phase
        phase = torch.tensor(np.pi, device=self.device)
        real, imag = ComplexTensorOps.complex_exp(phase)
        self.assertAlmostEqual(real.item(), -1.0, places=6)
        self.assertAlmostEqual(imag.item(), 0.0, places=6)
        
        # Test 2π phase
        phase = torch.tensor(2*np.pi, device=self.device)
        real, imag = ComplexTensorOps.complex_exp(phase)
        self.assertAlmostEqual(real.item(), 1.0, places=6)
        self.assertAlmostEqual(imag.item(), 0.0, places=6)
        
        # Test batch of phases
        phases = torch.tensor([0.0, np.pi/2, np.pi, 3*np.pi/2], device=self.device)
        real, imag = ComplexTensorOps.complex_exp(phases)
        expected_real = torch.tensor([1.0, 0.0, -1.0, 0.0], device=self.device)
        expected_imag = torch.tensor([0.0, 1.0, 0.0, -1.0], device=self.device)
        self.assertTrue(torch.allclose(real, expected_real, atol=1e-6))
        self.assertTrue(torch.allclose(imag, expected_imag, atol=1e-6))
        
        # Test gradient flow
        phase = torch.tensor(1.0, device=self.device, requires_grad=True)
        real, imag = ComplexTensorOps.complex_exp(phase)
        loss = real.sum() + imag.sum()
        loss.backward()
        self.assertIsNotNone(phase.grad)
        self.assertAlmostEqual(phase.grad.item(), -torch.sin(phase).item() + torch.cos(phase).item(), places=6)
        
        # Test with torch.autograd.gradcheck for a simple case
        def complex_exp_sum(x):
            real, imag = ComplexTensorOps.complex_exp(x)
            return real.sum() + imag.sum()
        
        x = torch.tensor([0.5], device=self.device, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(complex_exp_sum, x, eps=1e-6, atol=1e-4))
    
    def test_complex_mul(self):
        """Test complex multiplication function."""
        # Test multiplication of (1,0) by (1,0)
        a_real = torch.tensor(1.0, device=self.device)
        a_imag = torch.tensor(0.0, device=self.device)
        b_real = torch.tensor(1.0, device=self.device)
        b_imag = torch.tensor(0.0, device=self.device)
        
        real, imag = ComplexTensorOps.complex_mul(a_real, a_imag, b_real, b_imag)
        self.assertAlmostEqual(real.item(), 1.0, places=6)
        self.assertAlmostEqual(imag.item(), 0.0, places=6)
        
        # Test multiplication of (0,1) by (0,1)
        a_real = torch.tensor(0.0, device=self.device)
        a_imag = torch.tensor(1.0, device=self.device)
        b_real = torch.tensor(0.0, device=self.device)
        b_imag = torch.tensor(1.0, device=self.device)
        
        real, imag = ComplexTensorOps.complex_mul(a_real, a_imag, b_real, b_imag)
        self.assertAlmostEqual(real.item(), -1.0, places=6)
        self.assertAlmostEqual(imag.item(), 0.0, places=6)
        
        # Test multiplication of (1,1) by (1,1)
        a_real = torch.tensor(1.0, device=self.device)
        a_imag = torch.tensor(1.0, device=self.device)
        b_real = torch.tensor(1.0, device=self.device)
        b_imag = torch.tensor(1.0, device=self.device)
        
        real, imag = ComplexTensorOps.complex_mul(a_real, a_imag, b_real, b_imag)
        self.assertAlmostEqual(real.item(), 0.0, places=6)
        self.assertAlmostEqual(imag.item(), 2.0, places=6)
        
        # Test batch multiplication
        a_real = torch.tensor([1.0, 0.0, 1.0], device=self.device)
        a_imag = torch.tensor([0.0, 1.0, 1.0], device=self.device)
        b_real = torch.tensor([1.0, 0.0, 1.0], device=self.device)
        b_imag = torch.tensor([0.0, 1.0, 1.0], device=self.device)
        
        real, imag = ComplexTensorOps.complex_mul(a_real, a_imag, b_real, b_imag)
        expected_real = torch.tensor([1.0, -1.0, 0.0], device=self.device)
        expected_imag = torch.tensor([0.0, 0.0, 2.0], device=self.device)
        
        self.assertTrue(torch.allclose(real, expected_real, atol=1e-6))
        self.assertTrue(torch.allclose(imag, expected_imag, atol=1e-6))
        
        # Test broadcasting
        a_real = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        a_imag = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        b_real = torch.tensor(2.0, device=self.device)
        b_imag = torch.tensor(0.0, device=self.device)
        
        real, imag = ComplexTensorOps.complex_mul(a_real, a_imag, b_real, b_imag)
        expected_real = torch.tensor([2.0, 4.0, 6.0], device=self.device)
        expected_imag = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        
        self.assertTrue(torch.allclose(real, expected_real, atol=1e-6))
        self.assertTrue(torch.allclose(imag, expected_imag, atol=1e-6))
        
        # Test gradient flow
        a_real = torch.tensor(2.0, device=self.device, requires_grad=True)
        a_imag = torch.tensor(3.0, device=self.device, requires_grad=True)
        b_real = torch.tensor(4.0, device=self.device, requires_grad=True)
        b_imag = torch.tensor(5.0, device=self.device, requires_grad=True)
        
        real, imag = ComplexTensorOps.complex_mul(a_real, a_imag, b_real, b_imag)
        loss = real + imag
        loss.backward()
        
        self.assertIsNotNone(a_real.grad)
        self.assertIsNotNone(a_imag.grad)
        self.assertIsNotNone(b_real.grad)
        self.assertIsNotNone(b_imag.grad)
        
        # Check gradient values
        # For complex multiplication (a_real + i*a_imag) * (b_real + i*b_imag):
        # real = a_real*b_real - a_imag*b_imag
        # imag = a_real*b_imag + a_imag*b_real
        # 
        # Gradients:
        # ∂real/∂a_real = b_real, ∂real/∂a_imag = -b_imag
        # ∂real/∂b_real = a_real, ∂real/∂b_imag = -a_imag
        # ∂imag/∂a_real = b_imag, ∂imag/∂a_imag = b_real
        # ∂imag/∂b_real = a_imag, ∂imag/∂b_imag = a_real
        #
        # Since we're computing loss = real + imag, the gradients are:
        self.assertAlmostEqual(a_real.grad.item(), b_real.item() + b_imag.item(), places=6)
        self.assertAlmostEqual(a_imag.grad.item(), -b_imag.item() + b_real.item(), places=6)
        self.assertAlmostEqual(b_real.grad.item(), a_real.item() + a_imag.item(), places=6)
        self.assertAlmostEqual(b_imag.grad.item(), -a_imag.item() + a_real.item(), places=6)
    
    def test_complex_abs_squared(self):
        """Test complex absolute value squared function."""
        # Test abs_squared of (3,4)
        real = torch.tensor(3.0, device=self.device)
        imag = torch.tensor(4.0, device=self.device)
        
        abs_squared = ComplexTensorOps.complex_abs_squared(real, imag)
        self.assertAlmostEqual(abs_squared.item(), 25.0, places=6)
        
        # Test abs_squared of (0,0)
        real = torch.tensor(0.0, device=self.device)
        imag = torch.tensor(0.0, device=self.device)
        
        abs_squared = ComplexTensorOps.complex_abs_squared(real, imag)
        self.assertAlmostEqual(abs_squared.item(), 0.0, places=6)
        
        # Test batch operation
        real = torch.tensor([1.0, 3.0, 0.0], device=self.device)
        imag = torch.tensor([0.0, 4.0, 2.0], device=self.device)
        
        abs_squared = ComplexTensorOps.complex_abs_squared(real, imag)
        expected = torch.tensor([1.0, 25.0, 4.0], device=self.device)
        
        self.assertTrue(torch.allclose(abs_squared, expected, atol=1e-6))
        
        # Test gradient flow
        real = torch.tensor(3.0, device=self.device, requires_grad=True)
        imag = torch.tensor(4.0, device=self.device, requires_grad=True)
        
        abs_squared = ComplexTensorOps.complex_abs_squared(real, imag)
        abs_squared.backward()
        
        self.assertIsNotNone(real.grad)
        self.assertIsNotNone(imag.grad)
        
        # Check gradient values: d(r²+i²)/dr = 2r, d(r²+i²)/di = 2i
        self.assertAlmostEqual(real.grad.item(), 2 * real.item(), places=6)
        self.assertAlmostEqual(imag.grad.item(), 2 * imag.item(), places=6)
        
        # Test with large values
        real = torch.tensor(1e7, device=self.device)
        imag = torch.tensor(1e7, device=self.device)
        
        abs_squared = ComplexTensorOps.complex_abs_squared(real, imag)
        expected = real**2 + imag**2
        
        self.assertTrue(torch.isfinite(abs_squared))
        self.assertTrue(torch.allclose(abs_squared, expected, rtol=1e-5))
    
    def test_complex_exp_dwf(self):
        """Test complex exponential Debye-Waller factor function."""
        # Test zero displacement
        q_vec = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        u_vec = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        
        real, imag = ComplexTensorOps.complex_exp_dwf(q_vec, u_vec)
        self.assertAlmostEqual(real.item(), 1.0, places=6)
        self.assertAlmostEqual(imag.item(), 0.0, places=6)
        
        # Test increasing displacement decreases DWF exponentially
        u_values = [0.1, 0.2, 0.3, 0.4]
        dwf_values = []
        
        for u_val in u_values:
            u_vec = torch.tensor([u_val, u_val, u_val], device=self.device)
            real, _ = ComplexTensorOps.complex_exp_dwf(q_vec, u_vec)
            dwf_values.append(real.item())
        
        # Check that values decrease
        for i in range(1, len(dwf_values)):
            self.assertLess(dwf_values[i], dwf_values[i-1])
        
        # Test batch application
        q_vec = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=self.device)
        u_vec = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]], device=self.device)
        
        real, imag = ComplexTensorOps.complex_exp_dwf(q_vec, u_vec)
        
        # Expected: exp(-0.5 * q_i * u_i * q_i) for each batch element
        expected_real = torch.tensor([np.exp(-0.5 * 0.1), np.exp(-0.5 * 0.2)], device=self.device)
        expected_imag = torch.zeros_like(expected_real)
        
        self.assertTrue(torch.allclose(real, expected_real, atol=1e-6))
        self.assertTrue(torch.allclose(imag, expected_imag, atol=1e-6))
        
        # Test gradient flow
        q_vec = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        u_vec = torch.tensor([0.1, 0.2, 0.3], device=self.device, requires_grad=True)
        
        real, _ = ComplexTensorOps.complex_exp_dwf(q_vec, u_vec)
        real.backward()
        
        self.assertIsNotNone(u_vec.grad)
        
        # Check handling of large displacement values
        q_vec = torch.tensor([10.0, 10.0, 10.0], device=self.device)
        u_vec = torch.tensor([10.0, 10.0, 10.0], device=self.device)
        
        real, imag = ComplexTensorOps.complex_exp_dwf(q_vec, u_vec)
        
        self.assertTrue(torch.isfinite(real))
        self.assertTrue(torch.isfinite(imag))
        self.assertTrue(real > 0)  # DWF should always be positive
        self.assertTrue(torch.all(imag == 0))  # Imaginary part should be zero
    
    def test_complex_operations_gradient_flow(self):
        """Test gradient flow through a chain of complex operations."""
        # Create tensors with gradients
        a_real = torch.tensor(2.0, device=self.device, requires_grad=True)
        a_imag = torch.tensor(3.0, device=self.device, requires_grad=True)
        b_real = torch.tensor(4.0, device=self.device, requires_grad=True)
        b_imag = torch.tensor(5.0, device=self.device, requires_grad=True)
        phase = torch.tensor(0.5, device=self.device, requires_grad=True)
        
        # Create a simple network using complex operations
        # 1. Compute e^(i*phase)
        exp_real, exp_imag = ComplexTensorOps.complex_exp(phase)
        
        # 2. Multiply (a_real, a_imag) by (b_real, b_imag)
        mul_real, mul_imag = ComplexTensorOps.complex_mul(a_real, a_imag, b_real, b_imag)
        
        # 3. Multiply the results of steps 1 and 2
        final_real, final_imag = ComplexTensorOps.complex_mul(exp_real, exp_imag, mul_real, mul_imag)
        
        # 4. Compute the squared magnitude
        result = ComplexTensorOps.complex_abs_squared(final_real, final_imag)
        
        # Backpropagate
        result.backward()
        
        # Verify all gradients exist
        self.assertIsNotNone(a_real.grad)
        self.assertIsNotNone(a_imag.grad)
        self.assertIsNotNone(b_real.grad)
        self.assertIsNotNone(b_imag.grad)
        self.assertIsNotNone(phase.grad)
        
        # Verify gradients are finite
        self.assertTrue(torch.isfinite(a_real.grad))
        self.assertTrue(torch.isfinite(a_imag.grad))
        self.assertTrue(torch.isfinite(b_real.grad))
        self.assertTrue(torch.isfinite(b_imag.grad))
        self.assertTrue(torch.isfinite(phase.grad))


class TestEigenOps(unittest.TestCase):
    """Test cases for EigenOps class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set default device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Set default dtype
        torch.set_default_dtype(torch.float64)  # Use double precision for better accuracy in tests
    
    def test_svd_decomposition(self):
        """Test SVD decomposition function."""
        # Test with identity matrix
        matrix = torch.eye(3, device=self.device)
        U, S, V = EigenOps.svd_decomposition(matrix)
        
        # Check singular values
        expected_S = torch.ones(3, device=self.device)
        self.assertTrue(torch.allclose(S, expected_S, atol=1e-6))
        
        # Check reconstruction
        reconstructed = U @ torch.diag_embed(S) @ V.transpose(-2, -1)
        self.assertTrue(torch.allclose(reconstructed, matrix, atol=1e-6))
        
        # Test with rectangular matrix (tall)
        matrix = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], device=self.device)
        
        U, S, V = EigenOps.svd_decomposition(matrix)
        
        # Check shapes
        self.assertEqual(U.shape, (3, 2))
        self.assertEqual(S.shape, (2,))
        self.assertEqual(V.shape, (2, 2))
        
        # Check reconstruction
        reconstructed = U @ torch.diag_embed(S) @ V.transpose(-2, -1)
        self.assertTrue(torch.allclose(reconstructed, matrix, atol=1e-6))
        
        # Test with rectangular matrix (wide)
        matrix = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ], device=self.device)
        
        U, S, V = EigenOps.svd_decomposition(matrix)
        
        # Check shapes
        self.assertEqual(U.shape, (2, 2))
        self.assertEqual(S.shape, (2,))
        self.assertEqual(V.shape, (3, 2))
        
        # Check reconstruction
        reconstructed = U @ torch.diag_embed(S) @ V.transpose(-2, -1)
        self.assertTrue(torch.allclose(reconstructed, matrix, atol=1e-6))
        
        # Test with known singular values
        matrix = torch.diag(torch.tensor([5.0, 4.0, 3.0], device=self.device))
        U, S, V = EigenOps.svd_decomposition(matrix)
        
        expected_S = torch.tensor([5.0, 4.0, 3.0], device=self.device)
        self.assertTrue(torch.allclose(S, expected_S, atol=1e-6))
        
        # Test orthogonality of U and V
        matrix = torch.randn(4, 3, device=self.device)
        U, S, V = EigenOps.svd_decomposition(matrix)
        
        # U should have orthogonal columns
        U_orthogonal = torch.allclose(U.transpose(-2, -1) @ U, torch.eye(U.shape[-1], device=self.device), atol=1e-6)
        self.assertTrue(U_orthogonal)
        
        # V should have orthogonal columns
        V_orthogonal = torch.allclose(V.transpose(-2, -1) @ V, torch.eye(V.shape[-1], device=self.device), atol=1e-6)
        self.assertTrue(V_orthogonal)
        
        # Test with ill-conditioned matrix
        matrix = torch.tensor([
            [1.0, 1.0],
            [1.0, 1.0 + 1e-10]
        ], device=self.device)
        
        U, S, V = EigenOps.svd_decomposition(matrix)
        
        # Should not have NaN values
        self.assertFalse(torch.isnan(U).any())
        self.assertFalse(torch.isnan(S).any())
        self.assertFalse(torch.isnan(V).any())
        
        # Test with batched matrices
        matrices = torch.randn(2, 3, 3, device=self.device)
        U, S, V = EigenOps.svd_decomposition(matrices)
        
        # Check shapes
        self.assertEqual(U.shape, (2, 3, 3))
        self.assertEqual(S.shape, (2, 3))
        self.assertEqual(V.shape, (2, 3, 3))
        
        # Check reconstruction for each batch
        for i in range(matrices.shape[0]):
            reconstructed = U[i] @ torch.diag(S[i]) @ V[i].transpose(-2, -1)
            self.assertTrue(torch.allclose(reconstructed, matrices[i], atol=1e-6))
        
        # Test gradient flow
        matrix = torch.tensor([
            [2.0, 1.0],
            [1.0, 3.0]
        ], device=self.device, requires_grad=True)
        
        U, S, V = EigenOps.svd_decomposition(matrix)
        loss = S.sum()
        loss.backward()
        
        self.assertIsNotNone(matrix.grad)
        self.assertFalse(torch.isnan(matrix.grad).any())
    
    def test_eigen_decomposition(self):
        """Test eigendecomposition function."""
        # Test with identity matrix
        matrix = torch.eye(3, device=self.device)
        eigenvalues, eigenvectors = EigenOps.eigen_decomposition(matrix)
        
        # Check eigenvalues
        expected_eigenvalues = torch.ones(3, device=self.device)
        self.assertTrue(torch.allclose(eigenvalues, expected_eigenvalues, atol=1e-6))
        
        # Test with diagonal matrix
        matrix = torch.diag(torch.tensor([5.0, 4.0, 3.0], device=self.device))
        eigenvalues, eigenvectors = EigenOps.eigen_decomposition(matrix)
        
        # Eigenvalues should match diagonal (might be in different order)
        sorted_eigenvalues, _ = torch.sort(eigenvalues, descending=True)
        expected_eigenvalues = torch.tensor([5.0, 4.0, 3.0], device=self.device)
        self.assertTrue(torch.allclose(sorted_eigenvalues, expected_eigenvalues, atol=1e-6))
        
        # Test reconstruction
        reconstructed = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.transpose(-2, -1)
        self.assertTrue(torch.allclose(reconstructed, matrix, atol=1e-6))
        
        # Test with symmetric matrix
        matrix = torch.tensor([
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 2.0]
        ], device=self.device)
        
        eigenvalues, eigenvectors = EigenOps.eigen_decomposition(matrix)
        
        # Check reconstruction
        reconstructed = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.transpose(-2, -1)
        self.assertTrue(torch.allclose(reconstructed, matrix, atol=1e-6))
        
        # Test orthogonality of eigenvectors for symmetric matrix
        orthogonal = torch.allclose(
            eigenvectors.transpose(-2, -1) @ eigenvectors, 
            torch.eye(eigenvectors.shape[-1], device=self.device), 
            atol=1e-6
        )
        self.assertTrue(orthogonal)
        
        # Test with matrix having degenerate eigenvalues
        matrix = torch.tensor([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ], device=self.device)
        
        eigenvalues, eigenvectors = EigenOps.eigen_decomposition(matrix)
        
        # Should have eigenvalues [3, 2, 2] or [2, 2, 3] depending on sorting
        unique_eigenvalues = torch.unique(eigenvalues)
        self.assertEqual(len(unique_eigenvalues), 2)
        self.assertTrue((torch.abs(eigenvalues - 2.0) < 1e-5).any())
        self.assertTrue((torch.abs(eigenvalues - 3.0) < 1e-5).any())
        
        # Test with batched matrices
        matrices = torch.randn(2, 3, 3, device=self.device)
        # Make them symmetric
        matrices = matrices + matrices.transpose(-2, -1)
        
        eigenvalues, eigenvectors = EigenOps.eigen_decomposition(matrices)
        
        # Check shapes
        self.assertEqual(eigenvalues.shape, (2, 3))
        self.assertEqual(eigenvectors.shape, (2, 3, 3))
        
        # Check reconstruction for each batch
        for i in range(matrices.shape[0]):
            reconstructed = eigenvectors[i] @ torch.diag(eigenvalues[i]) @ eigenvectors[i].transpose(-2, -1)
            self.assertTrue(torch.allclose(reconstructed, matrices[i], atol=1e-6))
        
        # Test gradient flow
        matrix = torch.tensor([
            [2.0, 1.0],
            [1.0, 3.0]
        ], device=self.device, requires_grad=True)
        
        eigenvalues, eigenvectors = EigenOps.eigen_decomposition(matrix)
        loss = eigenvalues.sum()
        loss.backward()
        
        self.assertIsNotNone(matrix.grad)
        self.assertFalse(torch.isnan(matrix.grad).any())
    
    def test_solve_linear_system(self):
        """Test linear system solver function."""
        # Test with identity matrix
        A = torch.eye(3, device=self.device)
        b = torch.tensor([1.0, 2.0, 3.0], device=self.device).unsqueeze(-1)
        x = EigenOps.solve_linear_system(A, b)
        
        # Solution should match right-hand side
        self.assertTrue(torch.allclose(x, b, atol=1e-6))
        
        # Test with diagonal matrix
        A = torch.diag(torch.tensor([2.0, 3.0, 4.0], device=self.device))
        b = torch.tensor([4.0, 9.0, 16.0], device=self.device).unsqueeze(-1)
        x = EigenOps.solve_linear_system(A, b)
        
        # Expected solution: [2.0, 3.0, 4.0]
        expected_x = torch.tensor([2.0, 3.0, 4.0], device=self.device).unsqueeze(-1)
        self.assertTrue(torch.allclose(x, expected_x, atol=1e-6))
        
        # Test with overdetermined system (more equations than unknowns)
        A = torch.tensor([
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0]
        ], device=self.device)
        b = torch.tensor([2.0, 3.0, 4.0], device=self.device).unsqueeze(-1)
        x = EigenOps.solve_linear_system(A, b)
        
        # Check that Ax is approximately b (least squares solution)
        residual = torch.norm(A @ x - b)
        self.assertLess(residual, 1e-5)
        
        # Test with underdetermined system (more unknowns than equations)
        A = torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0]
        ], device=self.device)
        b = torch.tensor([2.0, 6.0], device=self.device).unsqueeze(-1)
        x = EigenOps.solve_linear_system(A, b)
        
        # Check that Ax = b
        self.assertTrue(torch.allclose(A @ x, b, atol=1e-6))
        
        # Test with ill-conditioned matrix
        A = torch.tensor([
            [1.0, 1.0],
            [1.0, 1.0 + 1e-10]
        ], device=self.device)
        b = torch.tensor([2.0, 2.0], device=self.device).unsqueeze(-1)
        x = EigenOps.solve_linear_system(A, b)
        
        # Should not have NaN values
        self.assertFalse(torch.isnan(x).any())
        
        # Test with batched matrices and multiple right-hand sides
        A = torch.randn(2, 3, 3, device=self.device)
        b = torch.randn(2, 3, 2, device=self.device)  # 2 batches, 3 equations, 2 right-hand sides
        x = EigenOps.solve_linear_system(A, b)
        
        # Check shapes
        self.assertEqual(x.shape, (2, 3, 2))
        
        # Check solution for each batch
        for i in range(A.shape[0]):
            self.assertTrue(torch.allclose(A[i] @ x[i], b[i], atol=1e-5))
        
        # Test gradient flow
        A = torch.tensor([
            [2.0, 1.0],
            [1.0, 3.0]
        ], device=self.device, requires_grad=True)
        b = torch.tensor([5.0, 8.0], device=self.device, requires_grad=True).unsqueeze(-1)
        
        # Ensure b is a leaf tensor
        b = b.detach().requires_grad_(True)
        
        x = EigenOps.solve_linear_system(A, b)
        loss = x.sum()
        loss.backward()
        
        self.assertIsNotNone(A.grad)
        self.assertIsNotNone(b.grad)
        self.assertFalse(torch.isnan(A.grad).any())
        self.assertFalse(torch.isnan(b.grad).any())


class TestGradientUtils(unittest.TestCase):
    """Test cases for GradientUtils class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set default device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Set default dtype
        torch.set_default_dtype(torch.float64)  # Use double precision for better accuracy in tests
    
    def test_finite_differences(self):
        """Test finite differences function."""
        # Test gradient of f(x) = x² at x=2
        def f1(x):
            return x**2
        
        x = torch.tensor(2.0, device=self.device)
        grad = GradientUtils.finite_differences(f1, x)
        
        # Expected gradient: 2x = 4
        self.assertAlmostEqual(grad.item(), 4.0, places=4)
        
        # Test gradient of f(x) = sin(x)
        def f2(x):
            return torch.sin(x)
        
        x = torch.tensor(np.pi/4, device=self.device)
        grad = GradientUtils.finite_differences(f2, x)
        
        # Expected gradient: cos(x) = cos(pi/4) = 1/sqrt(2)
        expected = np.cos(np.pi/4)
        self.assertAlmostEqual(grad.item(), expected, places=4)
        
        # Test with multidimensional input
        def f3(x):
            return torch.sum(x**2)
        
        x = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        grad = GradientUtils.finite_differences(f3, x)
        
        # Expected gradient: [2, 4, 6]
        expected = torch.tensor([2.0, 4.0, 6.0], device=self.device)
        self.assertTrue(torch.allclose(grad, expected, atol=1e-4))
        
        # Compare with torch.autograd.grad
        def f4(x):
            return torch.sum(x**3)
        
        x = torch.tensor([1.0, 2.0], device=self.device, requires_grad=True)
        y = f4(x)
        y.backward()
        autograd_result = x.grad
        
        x_no_grad = torch.tensor([1.0, 2.0], device=self.device)
        finite_diff_result = GradientUtils.finite_differences(f4, x_no_grad)
        
        # Expected gradient: [3x², 3x²] = [3, 12]
        self.assertTrue(torch.allclose(autograd_result, finite_diff_result, atol=1e-4))
        
        # Test with various step sizes
        step_sizes = [1e-2, 1e-4, 1e-6, 1e-8]
        results = []
        
        for eps in step_sizes:
            grad = GradientUtils.finite_differences(f1, torch.tensor(2.0, device=self.device), eps=eps)
            results.append(grad.item())
        
        # Results should converge to 4.0 as step size decreases (up to numerical precision)
        # Note: This test can be unstable due to floating point precision issues
        # We'll check that at least one result is close to 4.0
        self.assertTrue(any(abs(result - 4.0) < 1e-4 for result in results))
        
        # Test batch processing capability
        def f5(x):
            return torch.sum(x, dim=1)
        
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device)
        grad = GradientUtils.finite_differences(f5, x)
        
        # Expected gradient: [[1, 1], [1, 1]]
        expected = torch.ones_like(x)
        self.assertTrue(torch.allclose(grad, expected, atol=1e-4))
    
    def test_validate_gradients(self):
        """Test gradient validation function."""
        # Test with identical gradients
        analytical = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        numerical = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        
        valid, rel_errors, abs_errors = GradientUtils.validate_gradients(analytical, numerical)
        
        self.assertTrue(valid)
        self.assertTrue(torch.all(rel_errors == 0))
        self.assertTrue(torch.all(abs_errors == 0))
        
        # Test with slightly different gradients within tolerance
        analytical = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        numerical = torch.tensor([1.0001, 1.9998, 3.0002], device=self.device)
        
        valid, rel_errors, abs_errors = GradientUtils.validate_gradients(analytical, numerical)
        
        self.assertTrue(valid)
        self.assertTrue(torch.all(rel_errors <= 1e-4))
        self.assertTrue(torch.all(abs_errors <= 1e-4))
        
        # Test with gradients outside tolerance
        analytical = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        numerical = torch.tensor([1.1, 2.2, 3.3], device=self.device)
        
        valid, rel_errors, abs_errors = GradientUtils.validate_gradients(analytical, numerical)
        
        self.assertFalse(valid)
        
        # Test with zero gradients
        analytical = torch.zeros(3, device=self.device)
        numerical = torch.zeros(3, device=self.device)
        
        valid, rel_errors, abs_errors = GradientUtils.validate_gradients(analytical, numerical)
        
        self.assertTrue(valid)
        
        # Test with very large and very small gradients
        analytical = torch.tensor([1e-10, 1e10], device=self.device)
        numerical = torch.tensor([1.1e-10, 1.1e10], device=self.device)
        
        valid, rel_errors, abs_errors = GradientUtils.validate_gradients(analytical, numerical, rtol=0.2)
        
        self.assertTrue(valid)
        
        # Test with different tolerance values
        analytical = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        numerical = torch.tensor([1.05, 2.05, 3.05], device=self.device)
        
        # Should fail with default tolerance
        valid, _, _ = GradientUtils.validate_gradients(analytical, numerical)
        self.assertFalse(valid)
        
        # Should pass with higher tolerance
        valid, _, _ = GradientUtils.validate_gradients(analytical, numerical, rtol=0.1)
        self.assertTrue(valid)
    
    def test_gradient_norm(self):
        """Test gradient norm function."""
        # Test L2 norm of [3, 4]
        gradient = torch.tensor([3.0, 4.0], device=self.device)
        norm = GradientUtils.gradient_norm(gradient)
        
        self.assertAlmostEqual(norm.item(), 5.0, places=6)
        
        # Test L1 norm of [1, 2, 3]
        gradient = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        norm = GradientUtils.gradient_norm(gradient, ord=1)
        
        self.assertAlmostEqual(norm.item(), 6.0, places=6)
        
        # Test with higher-dimensional tensor
        gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device)
        norm = GradientUtils.gradient_norm(gradient)
        
        # Expected: sqrt(1² + 2² + 3² + 4²) = sqrt(30)
        self.assertAlmostEqual(norm.item(), np.sqrt(30), places=6)
        
        # Test with zero tensor
        gradient = torch.zeros(3, device=self.device)
        norm = GradientUtils.gradient_norm(gradient)
        
        self.assertAlmostEqual(norm.item(), 0.0, places=6)
        
        # Test with different norm orders
        gradient = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        
        # L1 norm
        norm_l1 = GradientUtils.gradient_norm(gradient, ord=1)
        self.assertAlmostEqual(norm_l1.item(), 6.0, places=6)
        
        # L2 norm
        norm_l2 = GradientUtils.gradient_norm(gradient, ord=2)
        self.assertAlmostEqual(norm_l2.item(), np.sqrt(14), places=6)
        
        # L-infinity norm
        norm_inf = GradientUtils.gradient_norm(gradient, ord=float('inf'))
        self.assertAlmostEqual(norm_inf.item(), 3.0, places=6)
        
        # Test with very large values
        gradient = torch.tensor([1e10, 2e10], device=self.device)
        norm = GradientUtils.gradient_norm(gradient)
        
        self.assertTrue(torch.isfinite(norm))
        self.assertAlmostEqual(norm.item() / 1e10, np.sqrt(5), places=6)


if __name__ == '__main__':
    unittest.main()
