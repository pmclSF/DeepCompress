from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

from .constants import LOG_2_RECIPROCAL


@dataclass
class TransformConfig:
    """Configuration for network transforms."""
    filters: int
    kernel_size: Tuple[int, int, int] = (3, 3, 3)
    strides: Tuple[int, int, int] = (1, 1, 1)
    activation: str = 'cenic_gdn'
    conv_type: str = 'separable'


class GDN(tf.keras.layers.Layer):
    """Generalized Divisive Normalization (Balle et al., 2016).

    y_i = x_i / sqrt(beta_i + sum_j(gamma_ij * x_j^2))

    When inverse=True, computes IGDN (inverse GDN) for the synthesis path:
    y_i = x_i * sqrt(beta_i + sum_j(gamma_ij * x_j^2))

    Args:
        inverse: If True, compute IGDN instead of GDN.
    """

    def __init__(self, inverse: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.inverse = inverse

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.beta = self.add_weight(
            name='beta',
            shape=[num_channels],
            initializer=tf.initializers.Ones(),
            constraint=tf.keras.constraints.NonNeg(),
            trainable=True
        )
        self.gamma = self.add_weight(
            name='gamma',
            shape=[num_channels, num_channels],
            initializer=tf.initializers.Identity(gain=0.1),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # Ensure gamma is non-negative and symmetric
        gamma = tf.nn.relu(self.gamma)
        gamma = (gamma + tf.transpose(gamma)) / 2.0

        # Compute normalization: beta_i + sum_j(gamma_ij * x_j^2)
        norm = tf.einsum('...c,cd->...d', inputs ** 2, gamma)
        norm = tf.sqrt(self.beta + norm)

        if self.inverse:
            return inputs * norm  # IGDN
        else:
            return inputs / norm  # GDN

    def get_config(self):
        config = super().get_config()
        config.update({
            'inverse': self.inverse
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

            if config.activation in ('gdn', 'cenic_gdn'):
                self.conv_layers.append(GDN(inverse=False))
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
            # Synthesis always needs Conv3DTranspose for upsampling
            # SpatialSeparableConv only supports forward (downsampling) convolution
            conv = tf.keras.layers.Conv3DTranspose(
                filters=current_filters,
                kernel_size=config.kernel_size,
                strides=config.strides,
                padding='same'
            )

            self.conv_layers.append(conv)

            if config.activation in ('gdn', 'cenic_gdn'):
                self.conv_layers.append(GDN(inverse=True))  # IGDN for synthesis
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

        # Final projection: outputs raw logits (no activation) for stable loss
        self.output_projection = tf.keras.layers.Conv3D(
            filters=1, kernel_size=(1, 1, 1), padding='same'
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

        # Add uniform noise for training, hard rounding for inference
        if training:
            y_hat = y + tf.random.uniform(tf.shape(y), -0.5, 0.5)
            z_noisy = z + tf.random.uniform(tf.shape(z), -0.5, 0.5)
        else:
            y_hat = tf.round(y)
            z_noisy = tf.round(z)

        # Synthesis — decode from quantized latent (y_hat), not raw encoder output
        z_hat = self.hyper_synthesis(z_noisy)
        x_hat = tf.sigmoid(self.output_projection(self.synthesis(y_hat)))

        return x_hat, y, z_hat, z_noisy

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

        # Final projection: outputs raw logits (no activation) for stable loss
        self.output_projection = tf.keras.layers.Conv3D(
            filters=1, kernel_size=(1, 1, 1), padding='same'
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

        # Compute latent channel dimensions dynamically from analysis transforms
        # Analysis doubles channels each block: filters -> 2*filters -> 4*filters
        self.latent_channels = config.filters * (2 ** 2)  # After 3 conv blocks with doubling
        self.hyper_channels = (config.filters // 2) * (2 ** 2)

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
                - x_hat: Reconstructed input (sigmoid of logits)
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
            # Original behavior with discretized likelihood
            if training:
                y_noisy = y + tf.random.uniform(tf.shape(y), -0.5, 0.5)
            else:
                y_noisy = tf.round(y)
            compressed, likelihood = self.entropy_module(y_noisy)
            y_hat = y_noisy
            y_bits = tf.reduce_sum(-tf.math.log(likelihood) * LOG_2_RECIPROCAL)
        else:
            # Advanced entropy models — pass z for hyper-latent rate
            y_hat, likelihood, y_bits = self.entropy_module(
                y, z_hat, z=z, training=training
            )

        # Compute z bits under learned prior
        if self.entropy_model_type == 'gaussian':
            # For gaussian, compute z bits directly
            if not hasattr(self, '_z_entropy') or not self._z_entropy.built:
                from .entropy_model import PatchedGaussianConditional
                self._z_entropy = PatchedGaussianConditional()
                self._z_entropy.build(z.shape)
            z_likelihood = self._z_entropy.likelihood(z)
            z_bits = tf.reduce_sum(-tf.math.log(z_likelihood) * LOG_2_RECIPROCAL)
        else:
            # z_bits already included in y_bits (via MeanScaleHyperprior)
            z_bits = tf.constant(0.0)

        total_bits = y_bits + z_bits

        # Synthesis — apply sigmoid to logits for output
        logits = self.output_projection(self.synthesis(y_hat))
        x_hat = tf.sigmoid(logits)

        # Rate information
        num_voxels = tf.cast(tf.reduce_prod(tf.shape(inputs)[1:4]), tf.float32)
        rate_info = {
            'likelihood': likelihood,
            'total_bits': total_bits,
            'y_bits': y_bits,
            'z_bits': z_bits,
            'bpp': total_bits / num_voxels,
            'logits': logits,
        }

        return x_hat, y, y_hat, z, rate_info

    def compress(self, inputs):
        """
        Compress inputs to bitstream representation.

        Args:
            inputs: Input voxel grid.

        Returns:
            Dict with compressed symbols and metadata.
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
        elif self.entropy_model_type in ('hyperprior', 'context'):
            compressed_y, side_info = self.entropy_module.compress(y, z_hat)
        elif self.entropy_model_type == 'channel':
            compressed_y, side_info = self.entropy_module.compress(y, z_hat)
        elif self.entropy_model_type in ('attention', 'hybrid'):
            # Attention/hybrid: use hyperprior mean for centered quantization
            # TODO: implement actual arithmetic coding for attention/hybrid models
            mean, scale = self.entropy_module.entropy_parameters(z_hat)
            compressed_y = tf.round(y - mean)
            side_info = {'mean': mean, 'scale': scale}
        else:
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
            Reconstructed voxel grid (sigmoid-applied probabilities).
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
        elif self.entropy_model_type in ('attention', 'hybrid'):
            # TODO: implement actual arithmetic coding for attention/hybrid models
            mean, _ = self.entropy_module.entropy_parameters(z_hat)
            y_hat = y_compressed + mean
        else:
            y_hat = y_compressed

        # Synthesis — apply sigmoid to logits
        x_hat = tf.sigmoid(self.output_projection(self.synthesis(y_hat)))

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
