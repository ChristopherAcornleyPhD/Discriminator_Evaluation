import tensorflow as tf
from classifiers import RecurrentClassifier
from layers.linear_replacement import LinearReplacement
from layers.empty_replacement import EmptyReplacement

class BiLSTMClassifier(RecurrentClassifier):
    def __init__(self):
        super(BiLSTMClassifier, self).__init__()
        self.recurrent = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))

class EmbeddingBiLSTM(BiLSTMClassifier):
    def __init__(self):
        super(EmbeddingBiLSTM, self).__init__()
        self.embedding = tf.keras.layers.Embedding(101, 64)

class LinearBiLSTM(BiLSTMClassifier):
    def __init__(self):
        super(LinearBiLSTM, self).__init__()
        self.embedding = LinearReplacement(64)

class EmptyBiLSTM(BiLSTMClassifier):
    def __init__(self):
        super(EmptyBiLSTM, self).__init__()
        self.embedding = EmptyReplacement()