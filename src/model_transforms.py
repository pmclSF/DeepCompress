import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv3D, Conv3DTranspose, AveragePooling3D, BatchNormalization, LayerNormalization
from tensorflow.keras.models import Sequential
from enum import Enum
from typing import List, Tuple, Callable, Dict
import logging

logger = logging.getLogger(__name__)

class SequentialLayer(Layer):
    def __init__(self, layers, *args, **kwargs):
        super(SequentialLayer, self).__init__(*args, **kwargs)
        self._layers = layers

    def call(self, tensor, **kwargs):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor

class ResidualLayer(Layer):
    def __init__(self, layers, residual_mode='add', data_format=None, *args, **kwargs):
        super(ResidualLayer, self).__init__(*args, **kwargs)
        assert residual_mode in ('add', 'concat')
        self._layers = layers
        self.residual_mode = residual_mode
        self.data_format = data_format

    def call(self, tensor, **kwargs):
        tensor = self._layers[0](tensor)
        residual = tensor
        for layer in self._layers[1:]:
            tensor = layer(tensor)
        if self.residual_mode == 'add':
            return residual + tensor
        else:
            return tf.concat([residual, tensor], axis=-1)

class AnalysisTransform(SequentialLayer):
    def __init__(self, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation=tf.nn.relu, *args, **kwargs):
        layers = [
            Conv3D(filters, kernel_size, strides=strides, padding='same', activation=activation),
            Conv3D(filters, kernel_size, padding='same', activation=activation),
            Conv3D(filters, kernel_size, padding='same', activation=None)
        ]
        super(AnalysisTransform, self).__init__(layers, *args, **kwargs)

class SynthesisTransform(SequentialLayer):
    def __init__(self, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation=tf.nn.relu, *args, **kwargs):
        layers = [
            Conv3DTranspose(filters, kernel_size, strides=strides, padding='same', activation=activation),
            Conv3DTranspose(filters, kernel_size, padding='same', activation=activation),
            Conv3DTranspose(1, kernel_size, padding='same', activation=activation)
        ]
        super(SynthesisTransform, self).__init__(layers, *args, **kwargs)

def custom_transform(filters: int, layer_specs: List[Tuple[str, Dict]], activation=tf.nn.relu):
    layers = []
    for layer_type, kwargs in layer_specs:
        if layer_type == "conv":
            layers.append(Conv3D(filters=filters, activation=activation, **kwargs))
        elif layer_type == "pool":
            layers.append(AveragePooling3D(**kwargs))
    return SequentialLayer(layers)

def latent_regularization(tensor: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.square(tensor))

def profile_transform(transform: Layer, input_shape: Tuple[int, int, int, int]):
    input_data = tf.random.uniform(input_shape)
    with tf.profiler.experimental.Profile('transform_profile'):
        _ = transform(input_data)

def normalization_layer(norm_type="batch") -> Layer:
    if norm_type == "batch":
        return BatchNormalization()
    elif norm_type == "instance":
        return LayerNormalization()

def experiment_with_activations(activations: List[Callable], transform_class: Callable, filters: int, input_shape: Tuple[int, int, int, int]):
    results = {}
    for activation in activations:
        transform = transform_class(filters=filters, activation=activation)
        input_data = tf.random.uniform(input_shape)
        output = transform(input_data)
        results[activation.__name__] = output.shape
    return results

def process_voxelized_input(transform: Layer, voxel_grid: tf.Tensor) -> tf.Tensor:
    """
    Process a voxelized input for point cloud compression.
    """
    return transform(voxel_grid)

def evaluate_transformation(predicted: tf.Tensor, target: tf.Tensor) -> Dict[str, float]:
    from src.pc_metric import calculate_chamfer_distance, calculate_d1_metric
    return {
        "Chamfer Distance": calculate_chamfer_distance(predicted.numpy(), target.numpy()),
        "D1 Metric": calculate_d1_metric(predicted.numpy(), target.numpy()),
    }

def advanced_activation(name="relu") -> Callable:
    activations = {
        "relu": tf.nn.relu,
        "swish": tf.nn.swish,
        "leaky_relu": tf.nn.leaky_relu
    }
    return activations.get(name, tf.nn.relu)

class TransformType(Enum):
    AnalysisTransform = AnalysisTransform
    SynthesisTransform = SynthesisTransform
