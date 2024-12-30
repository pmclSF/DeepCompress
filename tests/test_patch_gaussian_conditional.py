import sys
import os

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import tensorflow as tf
import numpy as np
from patch_gaussian_conditional import DummyGaussianConditional

def test_build():
    """Test the build functionality."""
    conditional = DummyGaussianConditional()
    conditional.build([16, 16])
    assert hasattr(conditional, '_quantized_cdf'), "Quantized CDF not built."
    assert hasattr(conditional, '_cdf_length'), "CDF length not built."
    assert hasattr(conditional, '_offset'), "Offset not built."

def test_decompress():
    """Test the decompress functionality."""
    conditional = DummyGaussianConditional()
    conditional.build([16, 16])

    strings = tf.constant(["test_string"], dtype=tf.string)
    scale = tf.constant([1.0], dtype=tf.float32)
    outputs = conditional.decompress(strings, scale)

    assert outputs.shape == tf.TensorShape([16, 16]), "Output shape mismatch."
    assert outputs.dtype == tf.float32, "Output dtype mismatch."

def test_prepare_indexes():
    """Test the preparation of indexes based on scale values."""
    conditional = DummyGaussianConditional()
    scale = tf.constant([0.5, 1.5, 2.5], dtype=tf.float32)
    indexes = conditional._prepare_indexes(scale)

    assert indexes.shape == scale.shape, "Indexes shape mismatch."
    assert tf.reduce_all(indexes >= 0), "Indexes contain invalid values."

def test_pmf_to_cdf():
    """Test the conversion of PMF to CDF."""
    conditional = DummyGaussianConditional()
    pmf = tf.random.uniform((3, 5), dtype=tf.float32)
    tail_mass = tf.constant([[0.01], [0.01], [0.01]], dtype=tf.float32)
    pmf_length = tf.constant([5, 5, 5], dtype=tf.int32)

    cdf = conditional._pmf_to_cdf(pmf, tail_mass, pmf_length, 5)

    assert cdf.shape == (3, 7), "CDF shape mismatch."
    assert tf.reduce_all(cdf >= 0), "CDF contains invalid values."

def test_standardized_cumulative():
    """Test the standardized cumulative distribution function."""
    conditional = DummyGaussianConditional()
    x = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32)
    result = conditional._standardized_cumulative(x)

    assert result.shape == x.shape, "Output shape mismatch."
    assert tf.reduce_all(result >= 0), "CDF values are invalid."
    assert tf.reduce_all(result <= 1), "CDF values are invalid."

def test_standardized_quantile():
    """Test the standardized quantile function."""
    conditional = DummyGaussianConditional()
    x = tf.constant([0.25, 0.5, 0.75], dtype=tf.float32)
    result = conditional._standardized_quantile(x)

    assert result.shape == x.shape, "Output shape mismatch."

def test_patch_gaussian_conditional():
    """Ensure patching adds the expected methods."""
    conditional = DummyGaussianConditional()
    assert hasattr(conditional, 'decompress'), "Decompress method not added."
    assert hasattr(conditional, 'build'), "Build method not added."

if __name__ == '__main__':
    test_build()
    test_decompress()
    test_prepare_indexes()
    test_pmf_to_cdf()
    test_standardized_cumulative()
    test_standardized_quantile()
    test_patch_gaussian_conditional()
