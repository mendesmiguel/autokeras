import numpy as np
from keras.layers import Dense, BatchNormalization, Activation

from autokeras.layers import WeightedAdd
from autokeras.utils import get_conv_layer_func, get_int_tuple


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


def wider_pre_conv(pre_layer, n_add_filters):
    pre_filter_shape = pre_layer.kernel_size
    n_pre_filters = pre_layer.filters
    rand = np.random.randint(n_pre_filters, size=n_add_filters)
    conv_func = get_conv_layer_func(len(pre_filter_shape))
    teacher_w, teacher_b = pre_layer.get_weights()
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
                              padding='same')
    new_pre_layer.build((None,) * (len(pre_filter_shape) + 1) + (student_w.shape[-2],))
    new_pre_layer.set_weights((student_w, student_b))
    return new_pre_layer


def wider_next_conv(layer, start_dim, total_dim, n_add):
    filter_shape = layer.kernel_size
    conv_func = get_conv_layer_func(len(filter_shape))
    n_filters = layer.filters
    teacher_w, teacher_b = layer.get_weights()

    new_weight_shape = list(teacher_w.shape)
    new_weight_shape[-2] = n_add
    new_weight = np.zeros(tuple(new_weight_shape))

    student_w = np.concatenate((teacher_w[..., :start_dim, :].copy(),
                                new_weight,
                                teacher_w[..., start_dim:total_dim, :].copy()), axis=-2)
    new_layer = conv_func(n_filters, kernel_size=filter_shape, padding='same')
    input_shape = list((None,) * (len(filter_shape) + 1) + (student_w.shape[-2],))
    new_layer.build(tuple(input_shape))
    new_layer.set_weights((student_w, teacher_b))
    return new_layer


def wider_bn(layer, start_dim, total_dim, n_add):
    weights = layer.get_weights()

    input_shape = list((None,) * layer.input_spec.ndim)
    input_shape[-1] = get_int_tuple(layer.gamma.shape)[0]
    input_shape[-1] += n_add

    temp_layer = BatchNormalization()
    add_input_shape = list(input_shape)
    add_input_shape[-1] = n_add
    temp_layer.build(tuple(add_input_shape))
    new_weights = temp_layer.get_weights()

    student_w = tuple()
    for weight, new_weight in zip(weights, new_weights):
        temp_w = weight.copy()
        temp_w = np.concatenate((temp_w[:start_dim], new_weight, temp_w[start_dim:total_dim]))
        student_w += (temp_w,)
    new_layer = BatchNormalization()
    new_layer.build(input_shape)
    new_layer.set_weights(student_w)
    return new_layer


def wider_next_dense(layer, start_dim, total_dim, n_add):
    n_units = layer.units
    teacher_w, teacher_b = layer.get_weights()
    student_w = teacher_w.copy()
    n_units_each_channel = int(teacher_w.shape[0] / total_dim)

    new_weight = np.zeros((n_add * n_units_each_channel, teacher_w.shape[1]))
    student_w = np.concatenate((student_w[:start_dim * n_units_each_channel],
                                new_weight,
                                student_w[start_dim * n_units_each_channel:total_dim * n_units_each_channel]))

    new_layer = Dense(n_units, activation=layer.get_config()['activation'])
    new_layer.build((None, student_w.shape[0]))
    new_layer.set_weights((student_w, teacher_b))
    return new_layer


def wider_weighted_add(layer, n_add):
    input_shape, _ = get_int_tuple(layer.input_shape)
    input_shape = list(input_shape)
    input_shape[-1] += n_add
    new_layer = WeightedAdd()
    # new_layer.build([input_shape, input_shape])
    new_layer.set_weights(layer.get_weights())
    return new_layer
