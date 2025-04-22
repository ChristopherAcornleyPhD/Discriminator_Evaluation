import tensorflow as tf

class DenseClassifier(tf.keras.Model):
    def __init__(self):
        super(DenseClassifier, self).__init__()
        self.embedding = None
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        x = self.pooling(x, training=training)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)
        return x
    
class ConvClassifier(tf.keras.Model):
    def __init__(self):
        super(ConvClassifier, self).__init__()
        self.embedding = None
        self.cnn = tf.keras.layers.Conv1D(32, 5, activation='relu')
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.dense2 = tf.keras.layers.Dense(64, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        x = self.cnn(x, training=training)
        x = self.pooling(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)
        return x
  
class RecurrentClassifier(tf.keras.Model):
    def __init__(self):
        super(RecurrentClassifier, self).__init__()
        self.embedding = None
        self.rnn = None
        self.dense2 = tf.keras.layers.Dense(64, activation='tanh') # should be relu
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        x = self.rnn(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)
        return x
    
class PoolingClassifier(tf.keras.Model):
    def __init__(self):
        super(PoolingClassifier, self).__init__()
        self.embedding = None
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.dense2 = tf.keras.layers.Dense(64, activation='tanh') # should be relu
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        x = self.pooling(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)
        return x