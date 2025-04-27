import tensorflow as tf

class MaxLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MaxLayer, self).__init__()
        self.beta = 1e10

    def call(self, inputs):
        x_range = tf.range(inputs.shape.as_list()[-1], dtype=inputs.dtype)
        return tf.reduce_sum(tf.nn.softmax(inputs*self.beta) * x_range, axis=-1)