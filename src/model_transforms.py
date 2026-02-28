from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

from .constants import EPSILON, LOG_2_RECIPROCAL


@dataclass
class TransformConfig:
    """Configuration for network transforms."""
    filters: int
    kernel_size: Tuple[int, int, int] = (3, 3, 3)
    strides: Tuple[int, int, int] = (1, 1, 1)
    activation: str = 'cenic_gdn'
    conv_type: str = 'separable'


class CENICGDN(tf.keras.layers.Layer):
    """CENIC-GDN activation function implementation."""

    def __init__(self, channels: int, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.beta = self.add_weight(
            name='beta',
            shape=[self.channels],
            initializer='ones',
            trainable=True
        )
        self.gamma = self.add_weight(
            name='gamma',
            shape=[self.channels, self.channels],
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # Note: XLA compilation removed as it breaks gradient flow when layers are composed
        norm = tf.abs(x)
        # Use axis 4 (channel dimension) for 5D tensors (batch, D, H, W, C)
        norm = tf.tensordot(norm, self.gamma, [[4], [0]])
        norm = tf.nn.bias_add(norm, self.beta)
        return x / tf.maximum(norm, EPSILON)

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels
        })
        return config


class SpatialSeparableConv(tf.keras.layers.Layer):
    """1+2D spatially separable convolution implementation."""

    def __init__(self,
                 filters: int,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 strides: Tuple[int, int, int] = (1, 1, 1),
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        # 1D path
        self.conv1d = tf.keras.layers.Conv3D(
            filters=filters // 2,
            kernel_size=(kernel_size[0], 1, 1),
            strides=(strides[0], 1, 1),
            padding='same'
        )

        # 2D path
        self.conv2d = tf.keras.layers.Conv3D(
            filters=filters,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            strides=(1, strides[1], strides[2]),
            padding='same'
        )

    def call(self, inputs):
        # Note: XLA compilation removed as it breaks gradient flow when layers are composed
        x = self.conv1d(inputs)
        return self.conv2d(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides
        })
        return config


class AnalysisTransform(tf.keras.layers.Layer):
    """Analysis transform with progressive channel expansion."""

    def __init__(self, config: TransformConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Define layers
        self.conv_layers = []
        current_filters = config.filters

        for i in range(3):  # Three blocks as per paper
            if config.conv_type == 'separable':
                conv = SpatialSeparableConv(
                    filters=current_filters,
                    kernel_size=config.kernel_size,
                    strides=config.strides
                )
            else:
                conv = tf.keras.layers.Conv3D(
                    filters=current_filters,
                    kernel_size=config.kernel_size,
                    strides=config.strides,
                    padding='same'
                )

            self.conv_layers.append(conv)

            if config.activation == 'cenic_gdn':
                self.conv_layers.append(CENICGDN(current_filters))
            else:
                self.conv_layers.append(tf.keras.layers.ReLU())

            current_filters *= 2  # Progressive channel expansion

    def call(self, inputs):
        # Note: XLA compilation removed as it breaks gradient flow when layers are composed
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'config': self.config
        })
        return config


class SynthesisTransform(tf.keras.layers.Layer):
    """Synthesis transform with progressive channel reduction."""

    def __init__(self, config: TransformConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Define layers
        self.conv_layers = []
        current_filters = config.filters * 4  # Start with max channels

        for i in range(3):  # Three blocks as per paper
            if config.conv_type == 'separable':
                conv = SpatialSeparableConv(
                    filters=current_filters,
                    kernel_size=config.kernel_size,
                    strides=config.strides
                )
            else:
                conv = tf.keras.layers.Conv3DTranspose(
                    filters=current_filters,
                    kernel_size=config.kernel_size,
                    strides=config.strides,
                    padding='same'
                )

            self.conv_layers.append(conv)

            if config.activation == 'cenic_gdn':
                self.conv_layers.append(CENICGDN(current_filters))
            else:
                self.conv_layers.append(tf.keras.layers.ReLU())

            current_filters = max(current_filters // 2, config.filters)  # Progressive reduction

    def call(self, inputs):
        # Note: XLA compilation removed as it breaks gradient flow when layers are composed
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'config': self.config
        })
        return config


class DeepCompressModel(tf.keras.Model):
    """Complete DeepCompress model implementation."""

    def __init__(self, config: TransformConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Main transforms
        self.analysis = AnalysisTransform(config)
        self.synthesis = SynthesisTransform(config)

        # Final projection: map from synthesis channels back to 1-channel occupancy
        self.output_projection = tf.keras.layers.Conv3D(
            filters=1, kernel_size=(1, 1, 1), activation='sigmoid', padding='same'
        )

        # Hyperprior
        self.hyper_analysis = AnalysisTransform(TransformConfig(
            filters=config.filters // 2,
            kernel_size=(1, 1, 1),
            activation='relu'
        ))
        self.hyper_synthesis = SynthesisTransform(TransformConfig(
            filters=config.filters // 2,
            kernel_size=(1, 1, 1),
            activation='relu'
        ))

    def call(self, inputs, training=None):
        # Analysis
        y = self.analysis(inputs)
        z = self.hyper_analysis(y)

        # Add uniform noise for training
        if training:
            y = y + tf.random.uniform(tf.shape(y), -0.5, 0.5)
            z = z + tf.random.uniform(tf.shape(z), -0.5, 0.5)

        # Synthesis
        y_hat = self.hyper_synthesis(z)
        x_hat = self.output_projection(self.synthesis(y))

        return x_hat, y, y_hat, z

    def get_config(self):
        config = super().get_config()
        config.update({
            'config': self.config
        })
        return config


class DeepCompressModelV2(tf.keras.Model):
    """
    Enhanced DeepCompress with configurable entropy model.

    Backward compatible - defaults to original behavior ('gaussian').
    Supports advanced entropy models for improved compression.

    Args:
        config: TransformConfig for network architecture.
        entropy_model: Type of entropy model to use:
            - 'gaussian': Original fixed Gaussian (backward compatible)
            - 'hyperprior': Mean-scale hyperprior
            - 'context': Autoregressive context model
            - 'channel': Channel-wise context model
            - 'attention': Attention-based context model
            - 'hybrid': Hybrid attention + channel context
        num_channel_groups: Number of groups for channel context models.
        num_attention_layers: Number of attention layers.
    """

    ENTROPY_MODELS = ['gaussian', 'hyperprior', 'context', 'channel', 'attention', 'hybrid']

    def __init__(self,
                 config: TransformConfig,
                 entropy_model: str = 'gaussian',
                 num_channel_groups: int = 4,
                 num_attention_layers: int = 2,
                 **kwargs):
        super().__init__(**kwargs)

        if entropy_model not in self.ENTROPY_MODELS:
            raise ValueError(
                f"entropy_model must be one of {self.ENTROPY_MODELS}, got '{entropy_model}'"
            )

        self.config = config
        self.entropy_model_type = entropy_model
        self.num_channel_groups = num_channel_groups
        self.num_attention_layers = num_attention_layers

        # Main transforms
        self.analysis = AnalysisTransform(config)
        self.synthesis = SynthesisTransform(config)

        # Final projection: map from synthesis channels back to 1-channel occupancy
        self.output_projection = tf.keras.layers.Conv3D(
            filters=1, kernel_size=(1, 1, 1), activation='sigmoid', padding='same'
        )

        # Hyperprior transforms
        self.hyper_analysis = AnalysisTransform(TransformConfig(
            filters=config.filters // 2,
            kernel_size=(1, 1, 1),
            activation='relu'
        ))
        self.hyper_synthesis = SynthesisTransform(TransformConfig(
            filters=config.filters // 2,
            kernel_size=(1, 1, 1),
            activation='relu'
        ))

        # Compute channel dimensions
        # Analysis progressively doubles channels 3 times
        self.latent_channels = config.filters * 4  # After 3 blocks of doubling
        self.hyper_channels = (config.filters // 2) * 4

        # Create entropy model based on selection
        self._create_entropy_model()

    def _create_entropy_model(self):
        """Create the selected entropy model."""
        if self.entropy_model_type == 'gaussian':
            from .entropy_model import EntropyModel
            self.entropy_module = EntropyModel()

        elif self.entropy_model_type == 'hyperprior':
            from .entropy_model import MeanScaleHyperprior
            self.entropy_module = MeanScaleHyperprior(
                latent_channels=self.latent_channels,
                hyper_channels=self.hyper_channels
            )

        elif self.entropy_model_type == 'context':
            from .context_model import ContextualEntropyModel
            self.entropy_module = ContextualEntropyModel(
                latent_channels=self.latent_channels,
                hyper_channels=self.hyper_channels
            )

        elif self.entropy_model_type == 'channel':
            from .channel_context import ChannelContextEntropyModel
            self.entropy_module = ChannelContextEntropyModel(
                latent_channels=self.latent_channels,
                hyper_channels=self.hyper_channels,
                num_groups=self.num_channel_groups
            )

        elif self.entropy_model_type == 'attention':
            from .attention_context import AttentionEntropyModel
            self.entropy_module = AttentionEntropyModel(
                latent_channels=self.latent_channels,
                hyper_channels=self.hyper_channels,
                num_attention_layers=self.num_attention_layers
            )

        elif self.entropy_model_type == 'hybrid':
            from .attention_context import HybridAttentionEntropyModel
            self.entropy_module = HybridAttentionEntropyModel(
                latent_channels=self.latent_channels,
                hyper_channels=self.hyper_channels,
                num_channel_groups=self.num_channel_groups,
                num_attention_layers=self.num_attention_layers
            )

    def call(self, inputs, training=None):
        """
        Forward pass through the model.

        Args:
            inputs: Input voxel grid.
            training: Whether in training mode.

        Returns:
            Tuple of (x_hat, y, y_hat, z, rate_info) where:
                - x_hat: Reconstructed input
                - y: Latent representation
                - y_hat: Quantized latent (or reconstructed)
                - z: Hyper-latent
                - rate_info: Dict with rate/likelihood information
        """
        # Analysis
        y = self.analysis(inputs)
        z = self.hyper_analysis(y)

        # Add uniform noise during training for z
        if training:
            z = z + tf.random.uniform(tf.shape(z), -0.5, 0.5)
        else:
            z = tf.round(z)

        # Hyper-synthesis
        z_hat = self.hyper_synthesis(z)

        # Entropy model processing
        if self.entropy_model_type == 'gaussian':
            # Original behavior
            if training:
                y_noisy = y + tf.random.uniform(tf.shape(y), -0.5, 0.5)
            else:
                y_noisy = tf.round(y)
            compressed, likelihood = self.entropy_module(y_noisy)
            y_hat = y_noisy
            # Using pre-computed reciprocal: multiplication is faster than division
            total_bits = -tf.reduce_sum(likelihood) * LOG_2_RECIPROCAL
        else:
            # Advanced entropy models
            y_hat, likelihood, total_bits = self.entropy_module(
                y, z_hat, training=training
            )

        # Synthesis
        x_hat = self.output_projection(self.synthesis(y_hat))

        # Rate information
        rate_info = {
            'likelihood': likelihood,
            'total_bits': total_bits,
            'bpp': total_bits / tf.cast(tf.reduce_prod(tf.shape(inputs)[1:4]), tf.float32)
        }

        return x_hat, y, y_hat, z, rate_info

    def compress(self, inputs):
        """
        Compress inputs to bitstream representation.

        Args:
            inputs: Input voxel grid.

        Returns:
            Tuple of (compressed_data, metadata) for storage/transmission.
        """
        # Analysis
        y = self.analysis(inputs)
        z = self.hyper_analysis(y)
        z_quantized = tf.round(z)

        # Hyper-synthesis for entropy model
        z_hat = self.hyper_synthesis(z_quantized)

        if self.entropy_model_type == 'gaussian':
            y_quantized = tf.round(y)
            compressed_y = y_quantized
            side_info = {}
        elif self.entropy_model_type in ['hyperprior', 'context']:
            compressed_y, side_info = self.entropy_module.compress(y, z_hat)
        elif self.entropy_model_type == 'channel':
            compressed_y, side_info = self.entropy_module.compress(y, z_hat)
        else:
            # For attention models, use basic quantization
            compressed_y = tf.round(y)
            side_info = {}

        return {
            'y': compressed_y,
            'z': z_quantized,
            'side_info': side_info
        }

    def decompress(self, compressed_data):
        """
        Decompress from bitstream representation.

        Args:
            compressed_data: Dict with compressed data from compress().

        Returns:
            Reconstructed voxel grid.
        """
        y_compressed = compressed_data['y']
        z = compressed_data['z']

        # Hyper-synthesis
        z_hat = self.hyper_synthesis(z)

        if self.entropy_model_type == 'gaussian':
            y_hat = y_compressed
        elif self.entropy_model_type == 'hyperprior':
            y_hat = self.entropy_module.decompress(y_compressed, z_hat)
        elif self.entropy_model_type == 'channel':
            y_hat = self.entropy_module.decode_parallel(z_hat, y_compressed)
        else:
            y_hat = y_compressed

        # Synthesis
        x_hat = self.output_projection(self.synthesis(y_hat))

        return x_hat

    def get_config(self):
        config = super().get_config()
        config.update({
            'config': self.config,
            'entropy_model': self.entropy_model_type,
            'num_channel_groups': self.num_channel_groups,
            'num_attention_layers': self.num_attention_layers
        })
        return config
