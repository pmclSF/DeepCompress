import tensorflow as tf
import tensorflow_probability as tfp
from typing import Optional, Dict, Any, Tuple

class PatchedGaussianConditional(tf.keras.layers.Layer):
    """Gaussian conditional layer with native TF 2.x operations."""
    
    def __init__(self, 
                 initial_scale: float = 1.0,
                 scale_table: Optional[tf.Tensor] = None,
                 tail_mass: float = 1e-9,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.initial_scale = initial_scale
        self.tail_mass = tail_mass
        
        if scale_table is not None:
            self.scale_table = tf.Variable(
                scale_table,
                trainable=False,
                name='scale_table'
            )
        else:
            self.scale_table = None
            
        self._debug_tensors = {}
        
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[1:],
            initializer=tf.keras.initializers.Constant(self.initial_scale),
            trainable=True
        )
        self.mean = self.add_weight(
            name='mean',
            shape=input_shape[1:],
            initializer='zeros',
            trainable=False
        )
        super().build(input_shape)
        
    @tf.function
    def quantize_scale(self, scale: tf.Tensor) -> tf.Tensor:
        if self.scale_table is None:
            return scale
            
        scale = tf.abs(scale)
        scale = tf.clip_by_value(
            scale,
            tf.reduce_min(self.scale_table),
            tf.reduce_max(self.scale_table)
        )
        
        scale_expanded = tf.expand_dims(scale, -1)
        table_expanded = tf.expand_dims(self.scale_table, 0)
        distances = tf.abs(scale_expanded - table_expanded)
        
        indices = tf.argmin(distances, axis=-1)
        return tf.gather(self.scale_table, indices)
        
    @tf.function
    def compress(self, inputs: tf.Tensor) -> tf.Tensor:
        scale = self.quantize_scale(self.scale)
        centered = inputs - self.mean
        normalized = centered / scale
        quantized = tf.round(normalized)
        
        self._debug_tensors.update({
            'compress_inputs': inputs,
            'compress_scale': scale,
            'compress_outputs': quantized
        })
        
        return quantized
        
    @tf.function
    def decompress(self, inputs: tf.Tensor) -> tf.Tensor:
        scale = self.quantize_scale(self.scale)
        denormalized = inputs * scale
        decompressed = denormalized + self.mean
        
        self._debug_tensors.update({
            'decompress_inputs': inputs,
            'decompress_scale': scale,
            'decompress_outputs': decompressed
        })
        
        return decompressed
        
    @tf.function
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        self._debug_tensors['inputs'] = inputs
        compressed = self.compress(inputs)
        outputs = self.decompress(compressed)
        self._debug_tensors['outputs'] = outputs
        return outputs
        
    def get_debug_tensors(self) -> Dict[str, tf.Tensor]:
        return self._debug_tensors
        
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'initial_scale': self.initial_scale,
            'scale_table': self.scale_table.numpy() if self.scale_table is not None else None,
            'tail_mass': self.tail_mass
        })
        return config

class EntropyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gaussian = PatchedGaussianConditional()

    def call(self, inputs, training=None):
        # Ensure the gaussian layer is built
        if not self.gaussian.built:
            self.gaussian.build(inputs.shape)

        compressed = self.gaussian.compress(inputs)
        likelihood = tfp.distributions.Normal(
            loc=self.gaussian.mean,
            scale=self.gaussian.scale
        ).log_prob(inputs)
        return compressed, likelihood


class ConditionalGaussian(tf.keras.layers.Layer):
    """
    Gaussian conditional that accepts external scale/mean parameters.

    Extends the basic Gaussian conditional with support for externally
    provided distribution parameters, enabling conditional entropy coding.

    Args:
        tail_mass: Tail mass for the distribution (default: 1e-9).
        scale_min: Minimum scale value to prevent numerical issues (default: 0.01).
    """

    def __init__(self,
                 tail_mass: float = 1e-9,
                 scale_min: float = 0.01,
                 **kwargs):
        super().__init__(**kwargs)
        self.tail_mass = tail_mass
        self.scale_min = scale_min
        self._debug_tensors = {}

    def _add_noise(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        """Add uniform noise during training for gradient estimation."""
        if training:
            noise = tf.random.uniform(tf.shape(inputs), -0.5, 0.5)
            return inputs + noise
        return tf.round(inputs)

    @tf.function
    def compress(self, inputs: tf.Tensor, scale: tf.Tensor, mean: tf.Tensor) -> tf.Tensor:
        """
        Compress inputs using provided scale and mean.

        Args:
            inputs: Input tensor to compress.
            scale: Scale parameter for the Gaussian distribution.
            mean: Mean parameter for the Gaussian distribution.

        Returns:
            Quantized (compressed) tensor.
        """
        # Ensure scale is positive
        scale = tf.maximum(scale, self.scale_min)

        # Center and normalize
        centered = inputs - mean
        quantized = tf.round(centered)

        self._debug_tensors.update({
            'compress_inputs': inputs,
            'compress_scale': scale,
            'compress_mean': mean,
            'compress_outputs': quantized
        })

        return quantized

    @tf.function
    def decompress(self, inputs: tf.Tensor, scale: tf.Tensor, mean: tf.Tensor) -> tf.Tensor:
        """
        Decompress inputs using provided scale and mean.

        Args:
            inputs: Quantized tensor to decompress.
            scale: Scale parameter for the Gaussian distribution.
            mean: Mean parameter for the Gaussian distribution.

        Returns:
            Decompressed (reconstructed) tensor.
        """
        # Add back the mean
        decompressed = inputs + mean

        self._debug_tensors.update({
            'decompress_inputs': inputs,
            'decompress_scale': scale,
            'decompress_mean': mean,
            'decompress_outputs': decompressed
        })

        return decompressed

    def call(self, inputs: tf.Tensor, scale: tf.Tensor, mean: tf.Tensor,
             training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Process inputs through compression and compute likelihood.

        Args:
            inputs: Input tensor.
            scale: Scale parameter tensor.
            mean: Mean parameter tensor.
            training: Whether in training mode.

        Returns:
            Tuple of (outputs, likelihood) where outputs are the reconstructed
            values and likelihood is the log-probability under the distribution.
        """
        self._debug_tensors['inputs'] = inputs

        # Ensure scale is positive
        scale = tf.maximum(scale, self.scale_min)

        # Center the input
        centered = inputs - mean

        # Quantize (round in eval, add noise in training)
        quantized = self._add_noise(centered, training=training or False)

        # Reconstruct
        outputs = quantized + mean

        # Compute likelihood using the Gaussian distribution
        distribution = tfp.distributions.Normal(loc=mean, scale=scale)
        likelihood = distribution.log_prob(inputs)

        self._debug_tensors['outputs'] = outputs
        self._debug_tensors['likelihood'] = likelihood

        return outputs, likelihood

    def get_debug_tensors(self) -> Dict[str, tf.Tensor]:
        return self._debug_tensors

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'tail_mass': self.tail_mass,
            'scale_min': self.scale_min
        })
        return config


class MeanScaleHyperprior(tf.keras.Model):
    """
    Complete mean-scale hyperprior entropy model.

    Combines EntropyParameters (for predicting mean/scale from hyperprior)
    with ConditionalGaussian (for entropy coding). This achieves better
    compression than fixed-parameter models by adapting the distribution
    to each spatial location based on learned hyperprior information.

    Args:
        latent_channels: Number of channels in the main latent representation.
        hyper_channels: Number of channels in the hyperprior representation.
        hidden_channels: Hidden channels for entropy parameters network.
    """

    def __init__(self,
                 latent_channels: int,
                 hyper_channels: int,
                 hidden_channels: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.latent_channels = latent_channels
        self.hyper_channels = hyper_channels
        self.hidden_channels = hidden_channels or latent_channels * 2

        # Import here to avoid circular dependency
        from entropy_parameters import EntropyParameters

        # Network to predict mean/scale from hyperprior
        self.entropy_parameters = EntropyParameters(
            latent_channels=latent_channels,
            hidden_channels=self.hidden_channels
        )

        # Conditional Gaussian for entropy coding
        self.conditional = ConditionalGaussian()

        # Hyperprior entropy model (for z)
        self.hyper_entropy = PatchedGaussianConditional()

    def call(self, y: tf.Tensor, z_hat: tf.Tensor,
             training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Process latent y using hyperprior z_hat.

        Args:
            y: Main latent representation.
            z_hat: Decoded hyperprior (typically from hyper_synthesis(z)).
            training: Whether in training mode.

        Returns:
            Tuple of (y_hat, y_likelihood, total_bits) where:
                - y_hat: Reconstructed latent
                - y_likelihood: Log-probability of y under the predicted distribution
                - total_bits: Estimated total bits for encoding
        """
        # Predict distribution parameters from hyperprior
        mean, scale = self.entropy_parameters(z_hat)

        # Process through conditional Gaussian
        y_hat, y_likelihood = self.conditional(y, scale, mean, training=training)

        # Estimate bits (negative log-likelihood converted to bits)
        bits_per_element = -y_likelihood / tf.math.log(2.0)
        total_bits = tf.reduce_sum(bits_per_element)

        return y_hat, y_likelihood, total_bits

    def compress(self, y: tf.Tensor, z_hat: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compress latent y using hyperprior z_hat.

        Args:
            y: Main latent representation.
            z_hat: Decoded hyperprior.

        Returns:
            Tuple of (symbols, side_info) where symbols are the quantized
            values and side_info contains mean/scale for decoding.
        """
        mean, scale = self.entropy_parameters(z_hat)
        symbols = self.conditional.compress(y, scale, mean)

        return symbols, {'mean': mean, 'scale': scale}

    def decompress(self, symbols: tf.Tensor, z_hat: tf.Tensor) -> tf.Tensor:
        """
        Decompress symbols using hyperprior z_hat.

        Args:
            symbols: Quantized symbols from compress().
            z_hat: Decoded hyperprior.

        Returns:
            Reconstructed latent y_hat.
        """
        mean, scale = self.entropy_parameters(z_hat)
        y_hat = self.conditional.decompress(symbols, scale, mean)

        return y_hat

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'latent_channels': self.latent_channels,
            'hyper_channels': self.hyper_channels,
            'hidden_channels': self.hidden_channels
        })
        return config