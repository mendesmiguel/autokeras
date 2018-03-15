from keras.layers import Conv3D, Conv2D, Conv1D, Dense

from autokeras.layers import StubDense, StubConv

CONV_FUNC_LIST = [Conv1D, Conv2D, Conv3D, StubConv]
WEIGHTED_LAYER_FUNC_LIST = CONV_FUNC_LIST + [Dense, StubDense]
MAX_MODEL_NUM = 1000
MAX_ITER_NUM = 200
MIN_LOSS_DEC = 1e-4
MAX_NO_IMPROVEMENT_NUM = 10
DEFAULT_SAVE_PATH = '/tmp/autokeras/'
EPOCHS_EACH = 1
N_NEIGHBORS = 8
ACQ_EXPLOITATION_DEPTH = 4

LAYER_ATTR = {'Dense': ['units', 'activation'],
              'Dropout': ['rate'],
              'MaxPooling1D': ['pool_size'],
              'MaxPooling2D': ['pool_size'],
              'MaxPooling3D': ['pool_size'],
              'Conv1D': ['filters', 'activation', 'kernel_size'],
              'Conv2D': ['filters', 'activation', 'kernel_size'],
              'Conv3D': ['filters', 'activation', 'kernel_size'],
              'Flatten': []
              }


