import tensorflow as tf

class LinearReplacement(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(LinearReplacement, self).__init__()
        self.dense_output = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs, training):
        outputs = tf.expand_dims(inputs, -1)
        return self.dense_output(outputs, training=training)
    

