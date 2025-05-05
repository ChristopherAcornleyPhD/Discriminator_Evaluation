import tensorflow as tf
from models.classifiers import RecurrentClassifier
from layers.linear_replacement import LinearReplacement
from layers.empty_replacement import EmptyReplacement

class GRUClassifier(RecurrentClassifier):
    def __init__(self, encoder):
        super(GRUClassifier, self).__init__(encoder)
        self.recurrent = tf.keras.layers.GRU(64)

class EmbeddingGRU(GRUClassifier):
    def __init__(self, vocal_size, encoder=None):
        super(EmbeddingGRU, self).__init__(encoder)
        self.embedding = tf.keras.layers.Embedding(vocal_size, 64)
        self.model_name = "GRU Embedding"

class LinearGRU(GRUClassifier):
    def __init__(self, encoder=None):
        super(LinearGRU, self).__init__(encoder)
        self.embedding = LinearReplacement(64)
        self.model_name = "GRU Linear"

class EmptyGRU(GRUClassifier):
    def __init__(self, encoder=None):
        super(EmptyGRU, self).__init__(encoder)
        self.embedding = EmptyReplacement()
        self.model_name = "GRU None"