import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

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
        
    @tf.function
    def call(self, x):
        norm = tf.abs(x)
        norm = tf.tensordot(norm, self.gamma, [[3], [0]])
        norm = tf.nn.bias_add(norm, self.beta)
        return x / norm

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

    @tf.function    
    def call(self, inputs):
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

    @tf.function
    def call(self, inputs):
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

    @tf.function
    def call(self, inputs):
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

    @tf.function
    def call(self, inputs):
        # Analysis
        y = self.analysis(inputs)
        z = self.hyper_analysis(y)
        
        # Add uniform noise for training
        if self.training:
            y = y + tf.random.uniform(tf.shape(y), -0.5, 0.5)
            z = z + tf.random.uniform(tf.shape(z), -0.5, 0.5)
        
        # Synthesis
        y_hat = self.hyper_synthesis(z)
        x_hat = self.synthesis(y)
        
        return x_hat, y, y_hat, z

    def get_config(self):
        config = super().get_config()
        config.update({
            'config': self.config
        })
        return config