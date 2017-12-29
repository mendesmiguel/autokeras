import os
from keras.layers import Conv1D, Conv2D, Conv3D
from tensorflow import Dimension

from autokeras.constant import CONV_FUNC_LIST, LAYER_ATTR
from autokeras import constant
from keras import backend


def is_conv_layer(layer):
    return isinstance(layer, tuple(CONV_FUNC_LIST))


def get_conv_layer_func(n_dim):
    conv_layer_functions = [Conv1D, Conv2D, Conv3D]
    if n_dim > 3:
        raise ValueError('The input dimension is too high.')
    if n_dim < 1:
        raise ValueError('The input dimension is too low.')
    return conv_layer_functions[n_dim - 1]


class ModelTrainer:
    def __init__(self, model, x_train, y_train, x_test, y_test, verbose):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.verbose = verbose
        self.training_losses = []
        self.minimum_loss = None
        self._no_improvement_count = 0

    def _converged(self, loss):
        self.training_losses.append(loss)
        if loss > (self.minimum_loss - constant.MIN_LOSS_DEC):
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0

        if loss < self.minimum_loss:
            self.minimum_loss = loss

        return self._no_improvement_count > constant.MAX_NO_IMPROVEMENT_NUM

    def train_model(self):
        self.training_losses = []
        self._no_improvement_count = 0
        self.minimum_loss = float('inf')
        for _ in range(constant.MAX_ITER_NUM):
            self.model.fit(self.x_train, self.y_train,
                           batch_size=min(self.x_train.shape[0], 200),
                           epochs=constant.EPOCHS_EACH,
                           verbose=self.verbose)
            loss, _ = self.model.evaluate(self.x_test, self.y_test, verbose=self.verbose)
            if self._converged(loss):
                break
        return self.minimum_loss


def copy_layer(layer, input_shape=None):
    if input_shape is None:
        input_shape = layer.input_shape
    new_layer = layer.__class__.from_config(layer.get_config())
    new_layer.build(input_shape)
    new_layer.set_weights(layer.get_weights())
    return new_layer


def extract_config(network):
    config = {'type': [], 'config': []}
    for layer in network.layers:
        name = type(layer).__name__
        config['type'].append(name)
        layer_config = layer.get_config()
        important_attr = {}
        for attr in LAYER_ATTR[name]:
            important_attr[attr] = layer_config[attr]
        config['config'].append(important_attr)
    return config


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_file_dir(path):
    ensure_dir(os.path.dirname(path))


def reset_weights(model):
    session = backend.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


def get_layer_size(layer):
    if is_conv_layer(layer):
        return layer.filters
    return layer.units


def get_int_tuple(temp_shape):
    input_shape = []
    for i in temp_shape:
        if isinstance(i, Dimension):
            input_shape.append(i.value)
        else:
            input_shape.append(i)
    return input_shape
