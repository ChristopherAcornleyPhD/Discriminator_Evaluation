import tensorflow as tf

class CoreClassifier(tf.keras.Model):
    def __init__(self):
        super(CoreClassifier, self).__init__()
        self.embedding = None
        self.dense_tanh = tf.keras.layers.Dense(64, activation='tanh')
        self.dense_output = tf.keras.layers.Dense(1)
        self.model_name = ""

class DenseClassifier(CoreClassifier):
    def __init__(self):
        super(DenseClassifier, self).__init__()
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.dense_relu = tf.keras.layers.Dense(128, activation='relu')

    def call(self, inputs, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        x = self.pooling(x, training=training)
        x = self.dense_relu(x, training=training)
        x = self.dense_tanh(x, training=training)
        x = self.dense_output(x, training=training)
        return x
    
class ConvClassifier(CoreClassifier):
    def __init__(self):
        super(ConvClassifier, self).__init__()
        self.cnn = tf.keras.layers.Conv1D(32, 5, activation='relu')
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        x = self.cnn(x, training=training)
        x = self.pooling(x, training=training)
        x = self.dense_tanh(x, training=training)
        x = self.dense_output(x, training=training)
        return x
  
class RecurrentClassifier(CoreClassifier):
    def __init__(self):
        super(RecurrentClassifier, self).__init__()
        self.recurrent = None

    def call(self, inputs, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        x = self.recurrent(x, training=training)
        x = self.dense_tanh(x, training=training)
        x = self.dense_output(x, training=training)
        return x
    
class PoolingClassifier(CoreClassifier):
    def __init__(self):
        super(PoolingClassifier, self).__init__()
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        x = self.pooling(x, training=training)
        x = self.dense_tanh(x, training=training)
        x = self.dense_output(x, training=training)
        return x