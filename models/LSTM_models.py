import tensorflow as tf
from classifiers import RecurrentClassifier
from layers.linear_replacement import LinearReplacement
from layers.empty_replacement import EmptyReplacement

class LSTMClassifier(RecurrentClassifier):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.recurrent = tf.keras.layers.LSTM(64)

class EmbeddingLSTM(LSTMClassifier):
    def __init__(self):
        super(EmbeddingLSTM, self).__init__()
        self.embedding = tf.keras.layers.Embedding(101, 64)

class LinearLSTM(LSTMClassifier):
    def __init__(self):
        super(LinearLSTM, self).__init__()
        self.embedding = LinearReplacement(64)

class EmptyLSTM(LSTMClassifier):
    def __init__(self):
        super(EmptyLSTM, self).__init__()
        self.embedding = EmptyReplacement()