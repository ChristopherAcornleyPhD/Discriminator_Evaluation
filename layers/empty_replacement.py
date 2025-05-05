import tensorflow as tf

class EmptyReplacement(tf.keras.layers.Layer):
    def __init__(self):
        super(EmptyReplacement, self).__init__()

    def call(self, inputs):
        return tf.cast(tf.expand_dims(inputs, -1), tf.float32)