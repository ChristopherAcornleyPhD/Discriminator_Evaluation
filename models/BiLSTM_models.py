import tensorflow as tf
from models.classifiers import RecurrentClassifier
from layers.linear_replacement import LinearReplacement
from layers.empty_replacement import EmptyReplacement

class BiLSTMClassifier(RecurrentClassifier):
    def __init__(self, encoder):
        super(BiLSTMClassifier, self).__init__(encoder)
        self.recurrent = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))

class EmbeddingBiLSTM(BiLSTMClassifier):
    def __init__(self, vocal_size, encoder=None):
        super(EmbeddingBiLSTM, self).__init__(encoder)
        self.embedding = tf.keras.layers.Embedding(vocal_size, 64)
        self.model_name = "Bi-LSTM Embedding"

class LinearBiLSTM(BiLSTMClassifier):
    def __init__(self, encoder=None):
        super(LinearBiLSTM, self).__init__(encoder)
        self.embedding = LinearReplacement(64)
        self.model_name = "Bi-LSTM Linear"

class EmptyBiLSTM(BiLSTMClassifier):
    def __init__(self, encoder=None):
        super(EmptyBiLSTM, self).__init__(encoder)
        self.embedding = EmptyReplacement()
        self.model_name = "Bi-LSTM None"