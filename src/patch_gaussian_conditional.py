import tensorflow as tf
from typing import Dict, Tuple, Optional

class PatchedGaussianConditional(tf.keras.layers.Layer):
    """A Gaussian conditional layer implemented with native TensorFlow operations."""
    
    def __init__(self, 
                 scale: tf.Tensor, 
                 mean: tf.Tensor,
                 scale_table: Optional[tf.Tensor] = None,
                 tail_mass: float = 1e-9,
                 **kwargs):
        """
        Initialize the Gaussian conditional layer.

        Args:
            scale: Scale parameter tensor of shape [H, W] or [B, H, W]
            mean: Mean parameter tensor of shape [H, W] or [B, H, W]
            scale_table: Optional quantization scale table of shape [N]
            tail_mass: Mass in the tails of the distribution
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)
        self.scale = tf.Variable(scale, trainable=True, name='scale')
        self.mean = tf.Variable(mean, trainable=False, name='mean')
        
        if scale_table is not None:
            self.scale_table = tf.Variable(
                scale_table, trainable=False, name='scale_table'
            )
        else:
            self.scale_table = None
            
        self.tail_mass = tail_mass
        self._debug_tensors = {}
        
    def quantize_scale(self, scale: tf.Tensor) -> tf.Tensor:
        """
        Quantize scale values using the scale table.
        
        Args:
            scale: Input scale tensor
            
        Returns:
            Quantized scale tensor
        """
        if self.scale_table is None:
            return scale
            
        # Ensure positive scales
        scale = tf.abs(scale)
        
        # Clip to scale table range
        scale = tf.clip_by_value(
            scale, 
            self.scale_table[0], 
            self.scale_table[-1]
        )
        
        # Find nearest neighbors in scale table
        scale_expanded = tf.expand_dims(scale, -1)
        table_expanded = tf.expand_dims(self.scale_table, 0)
        distances = tf.abs(scale_expanded - table_expanded)
        
        # Get indices of nearest scales
        indices = tf.argmin(distances, axis=-1)
        
        return tf.gather(self.scale_table, indices)
        
    def compress(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Compress inputs using quantization.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Quantized tensor
        """
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
        
    def decompress(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Decompress quantized values.
        
        Args:
            inputs: Quantized input tensor
            
        Returns:
            Decompressed tensor
        """
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
        
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass of the layer.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
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
        
    def get_config(self) -> Dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'scale': self.scale.numpy(),
            'mean': self.mean.numpy(),
            'scale_table': (
                self.scale_table.numpy() 
                if self.scale_table is not None 
                else None
            ),
            'tail_mass': self.tail_mass
        })
        return config