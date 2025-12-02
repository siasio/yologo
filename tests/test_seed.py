"""Tests for seed utilities."""

import random
import numpy as np
import torch
import pytest

from utils.seed import set_seed, get_random_state, set_random_state


def test_set_seed():
    """Test that set_seed produces reproducible results."""
    set_seed(42)

    # Generate some random numbers
    py_rand1 = random.random()
    np_rand1 = np.random.rand()
    torch_rand1 = torch.rand(1).item()

    # Reset seed
    set_seed(42)

    # Generate again - should be identical
    py_rand2 = random.random()
    np_rand2 = np.random.rand()
    torch_rand2 = torch.rand(1).item()

    assert py_rand1 == py_rand2
    assert np_rand1 == np_rand2
    assert torch_rand1 == torch_rand2


def test_set_seed_different_values():
    """Test that different seeds produce different results."""
    set_seed(42)
    rand1 = torch.rand(1).item()

    set_seed(123)
    rand2 = torch.rand(1).item()

    assert rand1 != rand2


def test_get_random_state():
    """Test capturing random state."""
    set_seed(42)

    state = get_random_state()

    assert "python" in state
    assert "numpy" in state
    assert "torch" in state


def test_set_random_state():
    """Test restoring random state."""
    set_seed(42)

    # Generate some numbers
    random.random()
    np.random.rand()
    torch.rand(1)

    # Capture state
    state = get_random_state()

    # Generate more numbers
    rand1 = random.random()
    rand2 = np.random.rand()
    rand3 = torch.rand(1).item()

    # Restore state
    set_random_state(state)

    # Generate again - should be identical to rand1, rand2, rand3
    rand1_restored = random.random()
    rand2_restored = np.random.rand()
    rand3_restored = torch.rand(1).item()

    assert rand1 == rand1_restored
    assert rand2 == rand2_restored
    assert rand3 == rand3_restored


def test_deterministic_mode():
    """Test deterministic mode setting."""
    # This mainly tests that the function runs without error
    set_seed(42, deterministic=True, benchmark=False)

    if torch.cuda.is_available():
        assert torch.backends.cudnn.deterministic == True
        assert torch.backends.cudnn.benchmark == False


def test_benchmark_mode():
    """Test benchmark mode setting."""
    set_seed(42, deterministic=False, benchmark=True)

    if torch.cuda.is_available():
        assert torch.backends.cudnn.benchmark == True
