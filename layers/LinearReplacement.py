import tensorflow as tf

class LinearReplacement(tf.keras.layers.Dense):
    def __init__(self, num_outputs):
        super(LinearReplacement, self).__init__(num_outputs)

    def call(self, inputs):
        outputs = tf.expand_dims(inputs, -1)
        return super(LinearReplacement, self).call(outputs)
    

