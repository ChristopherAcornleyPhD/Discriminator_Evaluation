import tensorflow as tf

class EmptyReplacement(tf.keras.layers.Dense):
    def __init__(self):
        super(EmptyReplacement, self).__init__()

    def call(self, inputs):
        outputs = tf.expand_dims(inputs, -1)
        return super(EmptyReplacement, self).call(outputs, training=False, use_bias=False)