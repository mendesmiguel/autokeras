from functools import reduce
from operator import mul

import numpy as np
from keras.layers import Dense, BatchNormalization, Activation

from autokeras.utils import get_conv_layer_func, is_conv_layer


def deeper_conv_block(conv_layer, kernel_size):
    filter_shape = (kernel_size,) * (len(conv_layer.kernel_size))
    n_filters = conv_layer.filters
    weight = np.zeros(filter_shape + (n_filters, n_filters))
    center = tuple(map(lambda x: int((x - 1) / 2), filter_shape))
    for i in range(n_filters):
        filter_weight = np.zeros(filter_shape + (n_filters,))
        index = center + (i,)
        filter_weight[index] = 1
        weight[..., i] = filter_weight
    bias = np.zeros(n_filters)
    conv_func = get_conv_layer_func(len(filter_shape))
    new_conv_layer = conv_func(n_filters,
                               kernel_size=filter_shape,
                               padding='same')
    new_conv_layer.build((None,) * (len(filter_shape) + 1) + (n_filters,))
    new_conv_layer.set_weights((weight, bias))
    return [new_conv_layer,
            BatchNormalization(),
            Activation('relu')]


def dense_to_deeper_layer(dense_layer):
    units = dense_layer.units
    weight = np.eye(units)
    bias = np.zeros(units)
    new_dense_layer = Dense(units, activation='relu')
    new_dense_layer.build((None, units))
    new_dense_layer.set_weights((weight, bias))
    return [new_dense_layer]


def dense_to_wider_layer(pre_layer, next_layer, n_add_units):
    n_units1 = pre_layer.get_weights()[0].shape[0]
    n_units2 = pre_layer.units
    n_units3 = next_layer.units

    teacher_w1 = pre_layer.get_weights()[0]
    teacher_b1 = pre_layer.get_weights()[1]
    teacher_w2 = next_layer.get_weights()[0]
    teacher_b2 = next_layer.get_weights()[1]
    rand = np.random.randint(n_units2, size=n_add_units)
    replication_factor = np.bincount(rand)
    student_w1 = teacher_w1.copy()
    student_w2 = teacher_w2.copy()
    student_b1 = teacher_b1.copy()

    # target layer update (i)
    for i in range(n_add_units):
        teacher_index = rand[i]
        new_weight = teacher_w1[:, teacher_index]
        new_weight = new_weight[:, np.newaxis]
        student_w1 = np.concatenate((student_w1, new_weight), axis=1)
        student_b1 = np.append(student_b1, teacher_b1[teacher_index])

    # next layer update (i+1)
    for i in range(n_add_units):
        teacher_index = rand[i]
        n_copies = replication_factor[teacher_index] + 1
        new_weight = teacher_w2[teacher_index, :] * (1. / n_copies)
        new_weight = new_weight[np.newaxis, :]
        student_w2 = np.concatenate((student_w2, new_weight), axis=0)
        student_w2[teacher_index, :] = new_weight

    new_pre_layer = Dense(n_units2 + n_add_units, input_shape=(n_units1,), activation='relu')
    new_pre_layer.build((None, n_units1))
    new_pre_layer.set_weights((student_w1, student_b1))
    new_next_layer = Dense(n_units3, activation=next_layer.get_config()['activation'])
    new_next_layer.build((None, n_units2 + n_add_units))
    new_next_layer.set_weights((student_w2, teacher_b2))

    return new_pre_layer, new_next_layer


def conv_to_wider_layer(pre_layer, next_layer_list, n_add_filters):
    # the next layer should be a list, return should be a layer and a list of layers.
    pre_filter_shape = pre_layer.kernel_size
    conv_func = get_conv_layer_func(len(pre_filter_shape))
    n_pre_filters = pre_layer.filters

    teacher_w, teacher_b = pre_layer.get_weights()

    rand = np.random.randint(n_pre_filters, size=n_add_filters)
    student_w = teacher_w.copy()
    student_b = teacher_b.copy()
    # target layer update (i)
    for i in range(len(rand)):
        teacher_index = rand[i]
        new_weight = teacher_w[..., teacher_index]
        new_weight = new_weight[..., np.newaxis]
        student_w = np.concatenate((student_w, new_weight), axis=-1)
        student_b = np.append(student_b, teacher_b[teacher_index])

    new_pre_layer = conv_func(n_pre_filters + n_add_filters,
                              kernel_size=pre_filter_shape,
                              padding='same',
                              input_shape=pre_layer.input_shape[1:])
    new_pre_layer.build((None,) * (len(pre_filter_shape) + 1) + (pre_layer.input_shape[-1],))
    new_pre_layer.set_weights((student_w, student_b))

    new_next_layer_list = []
    for next_layer in next_layer_list:
        input_shape = (None, ) * (len(pre_filter_shape) + 1) + (n_pre_filters + n_add_filters,)
        if is_conv_layer(next_layer):
            new_next_layer = wider_next_conv(input_shape, next_layer, rand)
        elif isinstance(next_layer, Dense):
            new_next_layer = wider_next_dense(n_pre_filters, next_layer, rand)
        else:
            new_next_layer = wider_next_bn(input_shape, next_layer, rand)
        new_next_layer_list.append(new_next_layer)

    return new_pre_layer, new_next_layer_list


def wider_next_conv(input_shape, next_layer, rand):
    replication_factor = np.bincount(rand)
    next_filter_shape = next_layer.kernel_size
    conv_func = get_conv_layer_func(len(next_filter_shape))
    n_next_filters = next_layer.filters
    teacher_w, teacher_b = next_layer.get_weights()
    student_w = teacher_w.copy()
    for i in range(len(rand)):
        teacher_index = rand[i]
        factor = replication_factor[teacher_index] + 1
        new_weight = teacher_w[..., teacher_index, :] * (1. / factor)
        new_weight_re = new_weight[..., np.newaxis, :]
        student_w = np.concatenate((student_w, new_weight_re), axis=-2)
        student_w[..., teacher_index, :] = new_weight
    new_next_layer = conv_func(n_next_filters, kernel_size=next_filter_shape, padding='same')
    new_next_layer.build(input_shape)
    new_next_layer.set_weights((student_w, teacher_b))
    return new_next_layer


def wider_next_bn(input_shape, next_layer, rand):
    weights = next_layer.get_weights()
    student_w = tuple()
    for weight in weights:
        temp_w = weight.copy()
        for i in range(len(rand)):
            temp_w = np.concatenate((temp_w, np.array([weight[rand[i]]])))
        student_w += (temp_w,)
    new_next_layer = BatchNormalization()
    new_next_layer.build(input_shape)
    new_next_layer.set_weights(student_w)
    return new_next_layer


def wider_next_dense(n_pre_filters, next_layer, rand):
    n_units = next_layer.units
    teacher_w, teacher_b = next_layer.get_weights()
    replication_factor = np.bincount(rand)
    n_total_weights = int(reduce(mul, teacher_w.shape))
    teacher_w = teacher_w.reshape(int(n_total_weights / n_pre_filters / n_units), n_pre_filters, n_units)
    student_w = teacher_w.copy()
    for i in range(len(rand)):
        teacher_index = rand[i]
        factor = replication_factor[teacher_index] + 1
        new_weight = teacher_w[:, teacher_index, :] * (1. / factor)
        new_weight_re = new_weight[:, np.newaxis, :]
        student_w = np.concatenate((student_w, new_weight_re), axis=1)
        student_w[:, teacher_index, :] = new_weight
    new_next_layer = Dense(n_units, activation=next_layer.get_config()['activation'])
    n_new_total_weights = int(reduce(mul, student_w.shape))
    input_dim = int(n_new_total_weights / n_units)
    new_next_layer.build((None, input_dim))
    new_next_layer.set_weights((student_w.reshape(input_dim, n_units), teacher_b))
    return new_next_layer
