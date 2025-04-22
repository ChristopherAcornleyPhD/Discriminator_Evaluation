import tensorflow as tf
from classifiers import RecurrentClassifier
from layers.LinearReplacement import LinearReplacement
from layers.EmptyReplacement import EmptyReplacement

class GRUClassifier(RecurrentClassifier):
    def __init__(self):
        super(GRUClassifier, self).__init__()
        self.recurrent = tf.keras.layers.GRU(64)

class EmbeddingGRU(GRUClassifier):
    def __init__(self):
        super(EmbeddingGRU, self).__init__()
        self.embedding = tf.keras.layers.Embedding(101, 64)

class LinearGRU(GRUClassifier):
    def __init__(self):
        super(LinearGRU, self).__init__()
        self.embedding = LinearReplacement(64)

class EmptyGRU(GRUClassifier):
    def __init__(self):
        super(EmptyGRU, self).__init__()
        self.embedding = EmptyReplacement()