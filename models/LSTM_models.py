import tensorflow as tf
from models.classifiers import RecurrentClassifier
from layers.linear_replacement import LinearReplacement
from layers.empty_replacement import EmptyReplacement

class LSTMClassifier(RecurrentClassifier):
    def __init__(self, encoder):
        super(LSTMClassifier, self).__init__(encoder)
        self.recurrent = tf.keras.layers.LSTM(64)

class EmbeddingLSTM(LSTMClassifier):
    def __init__(self, vocal_size, encoder=None):
        super(EmbeddingLSTM, self).__init__(encoder)
        self.embedding = tf.keras.layers.Embedding(vocal_size, 64)
        self.model_name = "LSTM Embedding"

class LinearLSTM(LSTMClassifier):
    def __init__(self, encoder=None):
        super(LinearLSTM, self).__init__(encoder)
        self.embedding = LinearReplacement(64)
        self.model_name = "LSTM Linear"

class EmptyLSTM(LSTMClassifier):
    def __init__(self, encoder=None):
        super(EmptyLSTM, self).__init__(encoder)
        self.embedding = EmptyReplacement()
        self.model_name = "LSTM None"