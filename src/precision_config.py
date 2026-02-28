"""
Mixed Precision Configuration for DeepCompress.

This module provides utilities for configuring mixed precision training,
which can provide ~50% memory reduction and 1.5-2x speedup on modern GPUs
with Tensor Cores (NVIDIA Volta, Turing, Ampere, and newer).

Usage:
    from precision_config import PrecisionManager

    # Enable mixed precision training
    PrecisionManager.configure('mixed_float16')

    # Wrap optimizer for loss scaling (required for float16)
    optimizer = PrecisionManager.wrap_optimizer(optimizer)

    # Check current compute dtype
    dtype = PrecisionManager.get_compute_dtype()
"""

import warnings
from typing import Optional

import tensorflow as tf


class PrecisionManager:
    """
    Manager for mixed precision training configuration.

    Mixed precision uses float16 for most computations (faster, less memory)
    while keeping critical operations in float32 (numerical stability).
    This is transparent to most model code.

    Supported policies:
    - 'float32': Default full precision (most compatible)
    - 'mixed_float16': Mixed precision for GPU training
    - 'mixed_bfloat16': Mixed precision for TPU/newer GPUs

    Important notes:
    - Entropy calculations (log probabilities) should remain in float32
    - Loss scaling is required for float16 gradient stability
    - Not all operations support float16 (some fall back automatically)
    """

    _original_policy: Optional[str] = None

    @classmethod
    def configure(cls, precision: str = 'float32', warn_on_cpu: bool = True) -> None:
        """
        Configure global mixed precision policy.

        Args:
            precision: One of 'float32', 'mixed_float16', or 'mixed_bfloat16'.
            warn_on_cpu: If True, warn when enabling float16 on CPU (no speedup).

        Raises:
            ValueError: If precision is not a valid policy name.
        """
        valid_policies = ['float32', 'mixed_float16', 'mixed_bfloat16']
        if precision not in valid_policies:
            raise ValueError(
                f"precision must be one of {valid_policies}, got '{precision}'"
            )

        # Store original policy for potential restoration
        cls._original_policy = tf.keras.mixed_precision.global_policy().name

        # Warn about CPU usage with float16
        if warn_on_cpu and precision in ['mixed_float16', 'mixed_bfloat16']:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                warnings.warn(
                    f"Enabling {precision} on CPU provides no speedup and may "
                    "be slower. Consider using 'float32' for CPU-only training.",
                    UserWarning
                )

        # Set the global policy
        policy = tf.keras.mixed_precision.Policy(precision)
        tf.keras.mixed_precision.set_global_policy(policy)

    @classmethod
    def restore_default(cls) -> None:
        """Restore the default float32 precision policy."""
        tf.keras.mixed_precision.set_global_policy('float32')
        cls._original_policy = None

    @classmethod
    def wrap_optimizer(
        cls,
        optimizer: tf.keras.optimizers.Optimizer,
        initial_scale: float = 2 ** 15,
        dynamic_growth_steps: int = 2000
    ) -> tf.keras.optimizers.Optimizer:
        """
        Wrap optimizer with loss scaling for mixed precision training.

        Loss scaling prevents gradient underflow in float16 by scaling
        the loss (and thus gradients) up during backprop, then scaling
        gradients back down before the weight update.

        Args:
            optimizer: The optimizer to wrap.
            initial_scale: Initial loss scale value (default: 2^15).
            dynamic_growth_steps: Steps between scale increases.

        Returns:
            The original optimizer if using float32, or a LossScaleOptimizer
            if using mixed precision.
        """
        policy = tf.keras.mixed_precision.global_policy()

        if policy.name in ['mixed_float16', 'mixed_bfloat16']:
            return tf.keras.mixed_precision.LossScaleOptimizer(
                optimizer,
                initial_scale=initial_scale,
                dynamic_growth_steps=dynamic_growth_steps
            )

        return optimizer

    @classmethod
    def get_compute_dtype(cls) -> tf.DType:
        """
        Get the current compute dtype from the global policy.

        Returns:
            tf.float16, tf.bfloat16, or tf.float32.
        """
        return tf.keras.mixed_precision.global_policy().compute_dtype

    @classmethod
    def get_variable_dtype(cls) -> tf.DType:
        """
        Get the current variable dtype from the global policy.

        Variables (weights) are typically kept in float32 even when
        compute dtype is float16 for numerical stability.

        Returns:
            Usually tf.float32.
        """
        return tf.keras.mixed_precision.global_policy().variable_dtype

    @classmethod
    def is_mixed_precision(cls) -> bool:
        """Check if mixed precision is currently enabled."""
        policy_name = tf.keras.mixed_precision.global_policy().name
        return policy_name in ['mixed_float16', 'mixed_bfloat16']

    @classmethod
    def cast_to_compute_dtype(cls, tensor: tf.Tensor) -> tf.Tensor:
        """
        Cast a tensor to the current compute dtype.

        Useful for ensuring input tensors match the expected precision.

        Args:
            tensor: Input tensor.

        Returns:
            Tensor cast to compute dtype.
        """
        return tf.cast(tensor, cls.get_compute_dtype())

    @classmethod
    def cast_to_float32(cls, tensor: tf.Tensor) -> tf.Tensor:
        """
        Cast a tensor to float32 for numerically sensitive operations.

        Use this for operations that require high precision, such as:
        - Log probability calculations
        - Softmax with large logits
        - Cumulative sums over long sequences

        Args:
            tensor: Input tensor.

        Returns:
            Tensor cast to float32.
        """
        return tf.cast(tensor, tf.float32)


def configure_for_gpu(enable_memory_growth: bool = True) -> None:
    """
    Configure TensorFlow for optimal GPU performance.

    This should be called before creating any tensors or models.

    Args:
        enable_memory_growth: If True, enable dynamic memory allocation
            instead of allocating all GPU memory upfront.
    """
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        return

    for gpu in gpus:
        if enable_memory_growth:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs are initialized
                warnings.warn(f"Could not set memory growth: {e}")


def get_recommended_precision() -> str:
    """
    Get the recommended precision policy for the current hardware.

    Returns:
        'mixed_float16' for NVIDIA GPUs with Tensor Cores,
        'mixed_bfloat16' for TPUs,
        'float32' otherwise.
    """
    # Check for TPU
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        if resolver:
            return 'mixed_bfloat16'
    except (ValueError, tf.errors.NotFoundError):
        pass

    # Check for GPU with compute capability >= 7.0 (Volta and newer)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Most modern GPUs support float16 well
        # Conservative: recommend only if GPU is available
        return 'mixed_float16'

    return 'float32'
