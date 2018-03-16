from keras import backend
from keras.layers import Add


class WeightedAdd(Add):
    """Weighted Add class inherited from Add class

    It's used to do add weights for data in Add layer

    Attributes:
        weights: backend variable
        one: const 1.0
        kernel: None
        _trainable_weights: list that store weight
    """
    def __init__(self, **kwargs):
        """Init Weighted add class"""
        super(WeightedAdd, self).__init__(**kwargs)
        self.weight = backend.variable(1.0)
        self.one = backend.constant(1.0)
        self.kernel = None
        self._trainable_weights.append(self.weight)

    def call(self, x, **kwargs):
        """Override call function in Add and return new weights"""
        a = backend.tf.scalar_mul(self.weight, x[0])
        b = backend.tf.scalar_mul(backend.tf.subtract(self.one, self.weight), x[1])
        c = backend.tf.add(a, b)
        return c

    def compute_output_shape(self, input_shape):
        """Return output_shape"""
        return input_shape


class StubLayer:
    def __init__(self, input_node=None, output_node=None):
        self.input = input_node
        self.output = output_node

    def build(self, shape):
        pass


class StubBatchNormalization(StubLayer):
    pass


class StubDense(StubLayer):
    def __init__(self, units, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.units = units
        self.output_shape = (None, units)


class StubConv(StubLayer):
    def __init__(self, filters, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.filters = filters
        self.output_shape = (None, filters)


class StubAggregateLayer(StubLayer):
    def __init__(self, input_nodes=None, output_node=None):
        if input_nodes is None:
            input_nodes = []
        super().__init__(input_nodes, output_node)


class StubConcatenate(StubAggregateLayer):
    pass


class StubWeightedAdd(StubAggregateLayer):
    pass


class StubActivation(StubLayer):
    pass


class StubPooling(StubLayer):
    pass


class StubInput(StubLayer):
    def __init__(self, units, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.units = units
        self.output_shape = (None, units)
