import tensorflow as tf
import tensorflow_probability as tfp
from typing import Optional, Dict, Any

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
        """Quantize scale values using the scale table."""
        if self.scale_table is None:
            return scale
            
        # Ensure positive scales
        scale = tf.abs(scale)
        
        # Clip to scale table range
        scale = tf.clip_by_value(
            scale,
            tf.reduce_min(self.scale_table),
            tf.reduce_max(self.scale_table)
        )
        
        # Find nearest neighbors in scale table
        scale_expanded = tf.expand_dims(scale, -1)
        table_expanded = tf.expand_dims(self.scale_table, 0)
        distances = tf.abs(scale_expanded - table_expanded)
        
        indices = tf.argmin(distances, axis=-1)
        return tf.gather(self.scale_table, indices)
        
    @tf.function
    def compress(self, inputs: tf.Tensor) -> tf.Tensor:
        """Compress inputs using quantization."""
        scale = self.quantize_scale(self.scale)
        
        # Center and normalize
        centered = inputs - self.mean
        normalized = centered / scale
        
        # Quantize to integers
        quantized = tf.round(normalized)
        
        self._debug_tensors.update({
            'compress_inputs': inputs,
            'compress_scale': scale,
            'compress_outputs': quantized
        })
        
        return quantized
        
    @tf.function
    def decompress(self, inputs: tf.Tensor) -> tf.Tensor:
        """Decompress quantized values."""
        scale = self.quantize_scale(self.scale)
        
        # Denormalize and decenter
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
        """Forward pass of the layer."""
        # Store original inputs
        self._debug_tensors['inputs'] = inputs
        
        # Compress and decompress
        compressed = self.compress(inputs)
        outputs = self.decompress(compressed)
        
        # Store outputs
        self._debug_tensors['outputs'] = outputs
        
        return outputs
        
    def get_debug_tensors(self) -> Dict[str, tf.Tensor]:
        """Get debug tensors dictionary."""
        return self._debug_tensors
        
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'initial_scale': self.initial_scale,
            'scale_table': self.scale_table.numpy() if self.scale_table is not None else None,
            'tail_mass': self.tail_mass
        })
        return config

class EntropyModel(tf.keras.Model):
    """Entropy model for compression."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gaussian = PatchedGaussianConditional()
        
    @tf.function
    def call(self, inputs, training=None):
        compressed = self.gaussian.compress(inputs)
        likelihood = tfp.distributions.Normal(
            loc=self.gaussian.mean,
            scale=self.gaussian.scale
        ).log_prob(inputs)
        return compressed, likelihood