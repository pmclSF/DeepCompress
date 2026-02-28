"""
Channel-wise Context Model for parallel-friendly entropy coding.

This module implements channel-wise autoregressive modeling where channels
are processed in groups. Previous channel groups provide context for
subsequent groups, enabling parallel decoding within each group while
maintaining autoregressive structure across groups.
"""

from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf

from .constants import LOG_2_RECIPROCAL


class SliceTransform(tf.keras.layers.Layer):
    """
    Transforms context from previous channel slices.

    Takes features from previously decoded channel groups and transforms
    them to provide contextual information for the current group.

    Args:
        in_channels: Number of input channels (from previous slices).
        out_channels: Number of output channels (for current slice parameters).
        hidden_channels: Number of hidden channels in transform.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels or out_channels

        # Transform layers
        self.conv1 = tf.keras.layers.Conv3D(
            filters=self.hidden_channels,
            kernel_size=3,
            padding='same',
            activation='relu',
            name='slice_conv1'
        )

        self.conv2 = tf.keras.layers.Conv3D(
            filters=self.hidden_channels,
            kernel_size=3,
            padding='same',
            activation='relu',
            name='slice_conv2'
        )

        self.conv_out = tf.keras.layers.Conv3D(
            filters=out_channels,
            kernel_size=1,
            padding='same',
            activation=None,
            name='slice_conv_out'
        )

    def call(self, context: tf.Tensor) -> tf.Tensor:
        """
        Transform context from previous channel slices.

        Args:
            context: Context tensor from previous channel groups.

        Returns:
            Transformed context for parameter prediction.
        """
        x = self.conv1(context)
        x = self.conv2(x)
        return self.conv_out(x)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'hidden_channels': self.hidden_channels
        })
        return config


class ChannelContext(tf.keras.layers.Layer):
    """
    Channel-wise autoregressive context model.

    Processes channels in groups, using previous groups as context.
    This is more parallel-friendly than spatial autoregressive because
    all positions within a channel group can be decoded simultaneously.

    Args:
        channels: Total number of latent channels.
        num_groups: Number of channel groups for autoregressive processing.
        hidden_channels: Hidden channels for slice transforms.
    """

    def __init__(self,
                 channels: int,
                 num_groups: int = 4,
                 hidden_channels: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.num_groups = num_groups
        self.hidden_channels = hidden_channels or channels

        if channels % num_groups != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_groups ({num_groups})"
            )

        self.channels_per_group = channels // num_groups

        # Create slice transforms for each group (except first)
        # Group i uses context from groups 0..i-1
        self.slice_transforms = []
        for i in range(1, num_groups):
            context_channels = i * self.channels_per_group
            out_channels = self.channels_per_group * 2  # mean and scale
            self.slice_transforms.append(
                SliceTransform(
                    in_channels=context_channels,
                    out_channels=out_channels,
                    hidden_channels=self.hidden_channels,
                    name=f'slice_transform_{i}'
                )
            )

    def call(self, y_hat: tf.Tensor, group_idx: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get context-based parameters for a specific channel group.

        Args:
            y_hat: Partially decoded latent (groups 0..group_idx-1 decoded).
            group_idx: Index of current group to process.

        Returns:
            Tuple of (context_mean, context_scale) for the current group.
            Returns zeros for the first group (no context available).
        """
        if group_idx == 0:
            # First group has no context
            batch_size = tf.shape(y_hat)[0]
            spatial_shape = tf.shape(y_hat)[1:4]
            zeros = tf.zeros(
                (batch_size, spatial_shape[0], spatial_shape[1],
                 spatial_shape[2], self.channels_per_group)
            )
            return zeros, zeros

        # Get context from previous groups
        context_end = group_idx * self.channels_per_group
        context = y_hat[..., :context_end]

        # Transform context to parameters
        params = self.slice_transforms[group_idx - 1](context)

        # Split into mean and scale
        mean, scale_raw = tf.split(params, 2, axis=-1)
        scale = tf.nn.softplus(scale_raw) + 0.01

        return mean, scale

    def get_all_context_params(self, y_hat: tf.Tensor) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        """
        Get context parameters for all groups (for training efficiency).

        Args:
            y_hat: Full decoded latent.

        Returns:
            List of (mean, scale) tuples for each group.
        """
        params_list = []
        for i in range(self.num_groups):
            mean, scale = self.call(y_hat, i)
            params_list.append((mean, scale))
        return params_list

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'num_groups': self.num_groups,
            'hidden_channels': self.hidden_channels
        })
        return config


class ChannelContextEntropyModel(tf.keras.Model):
    """
    Entropy model with channel-wise context.

    Combines hyperprior information with channel-wise autoregressive context
    for improved compression. Channels are split into groups and processed
    sequentially, with each group using previous groups as context.

    This enables:
    - Better compression than hyperprior alone
    - Parallel decoding within channel groups (faster than spatial AR)
    - Flexible trade-off between compression and speed via num_groups

    Args:
        latent_channels: Number of channels in the latent representation.
        hyper_channels: Number of channels in the hyperprior.
        num_groups: Number of channel groups (more groups = better compression, slower).
    """

    def __init__(self,
                 latent_channels: int,
                 hyper_channels: int,
                 num_groups: int = 4,
                 **kwargs):
        super().__init__(**kwargs)
        self.latent_channels = latent_channels
        self.hyper_channels = hyper_channels
        self.num_groups = num_groups

        if latent_channels % num_groups != 0:
            raise ValueError(
                f"latent_channels ({latent_channels}) must be divisible by "
                f"num_groups ({num_groups})"
            )

        self.channels_per_group = latent_channels // num_groups

        # Import here to avoid circular dependency
        from .entropy_model import ConditionalGaussian, PatchedGaussianConditional
        from .entropy_parameters import EntropyParameters

        # Hyperprior-based parameter prediction
        self.entropy_parameters = EntropyParameters(
            latent_channels=latent_channels
        )

        # Channel-wise context model
        self.channel_context = ChannelContext(
            channels=latent_channels,
            num_groups=num_groups
        )

        # Per-group conditional Gaussians
        self.conditionals = [
            ConditionalGaussian(name=f'conditional_{i}')
            for i in range(num_groups)
        ]

        # Hyperprior entropy model (for z)
        self.hyper_entropy = PatchedGaussianConditional()

        self.scale_min = 0.01

    def _fuse_params(self,
                     hyper_mean: tf.Tensor,
                     hyper_scale: tf.Tensor,
                     context_mean: tf.Tensor,
                     context_scale: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Fuse hyperprior and context parameters."""
        # Additive fusion for mean
        mean = hyper_mean + context_mean

        # Multiplicative fusion for scale (both contribute to uncertainty)
        scale = hyper_scale * (1.0 + context_scale)
        scale = tf.maximum(scale, self.scale_min)

        return mean, scale

    def call(self, y: tf.Tensor, z_hat: tf.Tensor,
             z: Optional[tf.Tensor] = None,
             training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Process latent y using hyperprior and channel-wise context.

        During training, uses ground truth y for all context (teacher forcing).
        During inference, processes groups sequentially.

        Args:
            y: Main latent representation.
            z_hat: Decoded hyperprior.
            z: Quantized/noised hyper-latent for computing z rate.
            training: Whether in training mode.

        Returns:
            Tuple of (y_hat, y_likelihood, total_bits).
        """
        # Get hyperprior-based parameters (full resolution)
        hyper_mean, hyper_scale = self.entropy_parameters(z_hat)

        y_hat_parts = []
        likelihood_parts = []

        # Process each channel group
        for i in range(self.num_groups):
            # Channel slice indices
            start_ch = i * self.channels_per_group
            end_ch = (i + 1) * self.channels_per_group

            # Get y slice for this group
            y_slice = y[..., start_ch:end_ch]

            # Get hyperprior params for this group
            hyper_mean_slice = hyper_mean[..., start_ch:end_ch]
            hyper_scale_slice = hyper_scale[..., start_ch:end_ch]

            # Get context params (using y for training, y_hat for inference)
            # Note: Use .call() to pass non-tensor group_idx as keyword argument
            if i == 0:
                # First group: no context available, channel_context returns zeros
                context_mean, context_scale = self.channel_context.call(y, group_idx=0)
            elif training:
                # Training: use ground truth y for context (teacher forcing)
                context_mean, context_scale = self.channel_context.call(y, group_idx=i)
            else:
                # Inference: use already decoded groups for context
                y_hat_partial = tf.concat(y_hat_parts, axis=-1)
                context_mean, context_scale = self.channel_context.call(
                    y_hat_partial, group_idx=i
                )

            # Fuse parameters
            mean, scale = self._fuse_params(
                hyper_mean_slice, hyper_scale_slice,
                context_mean, context_scale
            )

            # Process through conditional Gaussian
            y_hat_slice, likelihood_slice = self.conditionals[i](
                y_slice, scale, mean, training=training
            )

            y_hat_parts.append(y_hat_slice)
            likelihood_parts.append(likelihood_slice)

        # Concatenate all groups
        y_hat = tf.concat(y_hat_parts, axis=-1)
        y_likelihood = tf.concat(likelihood_parts, axis=-1)

        # Compute y bits from discretized likelihood
        y_bits = tf.reduce_sum(-tf.math.log(y_likelihood) * LOG_2_RECIPROCAL)

        # Compute z bits if z is provided
        z_bits = tf.constant(0.0)
        if z is not None:
            if not self.hyper_entropy.built:
                self.hyper_entropy.build(z.shape)
            z_likelihood = self.hyper_entropy.likelihood(z)
            z_bits = tf.reduce_sum(-tf.math.log(z_likelihood) * LOG_2_RECIPROCAL)

        total_bits = y_bits + z_bits

        return y_hat, y_likelihood, total_bits

    def decode_parallel(self, z_hat: tf.Tensor, symbols: tf.Tensor) -> tf.Tensor:
        """
        Parallel decoding within channel groups.

        This is faster than full sequential decoding because each
        channel group can be decoded in parallel.

        Args:
            z_hat: Decoded hyperprior.
            symbols: Quantized symbols (from encoder).

        Returns:
            Reconstructed y_hat.
        """
        # Get hyperprior parameters
        hyper_mean, hyper_scale = self.entropy_parameters(z_hat)

        y_hat_parts = []

        for i in range(self.num_groups):
            start_ch = i * self.channels_per_group
            end_ch = (i + 1) * self.channels_per_group

            # Get symbol slice
            symbols_slice = symbols[..., start_ch:end_ch]

            # Get hyperprior params
            hyper_mean_slice = hyper_mean[..., start_ch:end_ch]
            hyper_scale_slice = hyper_scale[..., start_ch:end_ch]

            # Get context from previous groups (optimized: no padding needed!)
            # The channel_context only accesses channels 0..group_idx-1, so we
            # avoid creating unnecessary zero-padded tensors.
            if i == 0:
                # First group: no context needed, channel_context returns zeros
                y_hat_partial = symbols  # Just for shape reference
            else:
                # Only concatenate decoded parts - no padding
                y_hat_partial = tf.concat(y_hat_parts, axis=-1)

            context_mean, context_scale = self.channel_context.call(y_hat_partial, group_idx=i)

            # Fuse parameters
            mean, scale = self._fuse_params(
                hyper_mean_slice, hyper_scale_slice,
                context_mean, context_scale
            )

            # Decompress (add mean back)
            y_hat_slice = self.conditionals[i].decompress(symbols_slice, scale, mean)
            y_hat_parts.append(y_hat_slice)

        return tf.concat(y_hat_parts, axis=-1)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'latent_channels': self.latent_channels,
            'hyper_channels': self.hyper_channels,
            'num_groups': self.num_groups
        })
        return config
