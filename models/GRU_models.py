import tensorflow as tf
from models.classifiers import RecurrentClassifier
from layers.linear_replacement import LinearReplacement
from layers.empty_replacement import EmptyReplacement

class GRUClassifier(RecurrentClassifier):
    def __init__(self):
        super(GRUClassifier, self).__init__()
        self.recurrent = tf.keras.layers.GRU(64)

class EmbeddingGRU(GRUClassifier):
    def __init__(self):
        super(EmbeddingGRU, self).__init__()
        self.embedding = tf.keras.layers.Embedding(101, 64)
        self.model_name = "GRU Embedding"

class LinearGRU(GRUClassifier):
    def __init__(self):
        super(LinearGRU, self).__init__()
        self.embedding = LinearReplacement(64)
        self.model_name = "GRU Linear"

class EmptyGRU(GRUClassifier):
    def __init__(self):
        super(EmptyGRU, self).__init__()
        self.embedding = EmptyReplacement()
        self.model_name = "GRU None"