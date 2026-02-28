"""
Entropy Parameters Network for Mean-Scale Hyperprior.

This module implements networks that predict Gaussian distribution parameters
(mean, scale) from hyperprior information, enabling conditional entropy coding
where the distribution is adapted based on learned side information.
"""

from typing import Optional, Tuple

import tensorflow as tf


class EntropyParameters(tf.keras.layers.Layer):
    """
    Predicts Gaussian distribution parameters (mean, scale) from hyperprior.

    This enables conditional entropy coding where the distribution is
    adapted based on learned side information. The network takes the
    decoded hyperprior z_hat and outputs mean and scale parameters
    for the latent distribution.

    Args:
        latent_channels: Number of channels in the latent representation.
        hidden_channels: Number of hidden channels (default: 2x latent_channels).
        num_layers: Number of convolutional layers (default: 3).
        kernel_size: Kernel size for convolutions (default: 3).
    """

    def __init__(self,
                 latent_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_layers: int = 3,
                 kernel_size: int = 3,
                 **kwargs):
        super().__init__(**kwargs)
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels or latent_channels * 2
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        # Build shared feature extraction layers
        self.feature_layers = []
        for i in range(num_layers - 1):
            self.feature_layers.append(
                tf.keras.layers.Conv3D(
                    filters=self.hidden_channels,
                    kernel_size=kernel_size,
                    padding='same',
                    activation='relu',
                    name=f'feature_conv_{i}'
                )
            )

        # Output layers for mean and scale
        # Mean can be any value
        self.mean_conv = tf.keras.layers.Conv3D(
            filters=latent_channels,
            kernel_size=kernel_size,
            padding='same',
            activation=None,
            name='mean_conv'
        )

        # Scale must be positive, using softplus activation
        self.scale_conv = tf.keras.layers.Conv3D(
            filters=latent_channels,
            kernel_size=kernel_size,
            padding='same',
            activation=None,
            name='scale_conv'
        )

        # Minimum scale to avoid numerical issues
        self.scale_min = 0.01

    def call(self, z_hat: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict mean and scale from hyperprior.

        Args:
            z_hat: Decoded hyperprior tensor of shape (B, D, H, W, C_hyper).

        Returns:
            Tuple of (mean, scale) tensors, each of shape (B, D, H, W, C_latent).
        """
        # Shared feature extraction
        x = z_hat
        for layer in self.feature_layers:
            x = layer(x)

        # Predict mean (no activation)
        mean = self.mean_conv(x)

        # Predict scale (softplus ensures positive values)
        scale_raw = self.scale_conv(x)
        scale = tf.nn.softplus(scale_raw) + self.scale_min

        return mean, scale

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_channels': self.latent_channels,
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'kernel_size': self.kernel_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EntropyParametersWithContext(tf.keras.layers.Layer):
    """
    Predicts Gaussian parameters from both hyperprior and local context.

    Combines hyperprior information with spatial context from neighboring
    decoded symbols for more accurate distribution prediction.

    Args:
        latent_channels: Number of channels in the latent representation.
        context_channels: Number of context input channels.
        hidden_channels: Number of hidden channels.
    """

    def __init__(self,
                 latent_channels: int,
                 context_channels: int,
                 hidden_channels: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.latent_channels = latent_channels
        self.context_channels = context_channels
        self.hidden_channels = hidden_channels or latent_channels * 2

        # Hyperprior processing branch
        self.hyper_conv = tf.keras.layers.Conv3D(
            filters=self.hidden_channels,
            kernel_size=3,
            padding='same',
            activation='relu',
            name='hyper_conv'
        )

        # Context processing branch
        self.context_conv = tf.keras.layers.Conv3D(
            filters=self.hidden_channels,
            kernel_size=3,
            padding='same',
            activation='relu',
            name='context_conv'
        )

        # Fusion layers
        self.fusion_conv = tf.keras.layers.Conv3D(
            filters=self.hidden_channels,
            kernel_size=1,
            padding='same',
            activation='relu',
            name='fusion_conv'
        )

        # Output heads
        self.mean_conv = tf.keras.layers.Conv3D(
            filters=latent_channels,
            kernel_size=1,
            padding='same',
            activation=None,
            name='mean_conv'
        )

        self.scale_conv = tf.keras.layers.Conv3D(
            filters=latent_channels,
            kernel_size=1,
            padding='same',
            activation=None,
            name='scale_conv'
        )

        self.scale_min = 0.01

    def call(self, z_hat: tf.Tensor, context: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict mean and scale from hyperprior and context.

        Args:
            z_hat: Decoded hyperprior tensor.
            context: Context tensor from previously decoded symbols.

        Returns:
            Tuple of (mean, scale) tensors.
        """
        # Process hyperprior
        hyper_features = self.hyper_conv(z_hat)

        # Process context
        context_features = self.context_conv(context)

        # Fuse features
        combined = tf.concat([hyper_features, context_features], axis=-1)
        fused = self.fusion_conv(combined)

        # Predict parameters
        mean = self.mean_conv(fused)
        scale_raw = self.scale_conv(fused)
        scale = tf.nn.softplus(scale_raw) + self.scale_min

        return mean, scale

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_channels': self.latent_channels,
            'context_channels': self.context_channels,
            'hidden_channels': self.hidden_channels
        })
        return config
