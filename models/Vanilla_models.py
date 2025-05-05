import tensorflow as tf
from models.classifiers import RecurrentClassifier
from layers.linear_replacement import LinearReplacement
from layers.empty_replacement import EmptyReplacement

class VanillaClassifier(RecurrentClassifier):
    def __init__(self, encoder):
        super(VanillaClassifier, self).__init__(encoder)
        self.recurrent = tf.keras.layers.SimpleRNN(64)

class EmbeddingVanilla(VanillaClassifier):
    def __init__(self, vocal_size, encoder=None):
        super(EmbeddingVanilla, self).__init__(encoder)
        self.embedding = tf.keras.layers.Embedding(vocal_size, 64)
        self.model_name = "Vanilla Embedding"

class LinearVanilla(VanillaClassifier):
    def __init__(self, encoder=None):
        super(LinearVanilla, self).__init__(encoder)
        self.embedding = LinearReplacement(64)
        self.model_name = "Vanilla Linear"

class EmptyVanilla(VanillaClassifier):
    def __init__(self, encoder=None):
        super(EmptyVanilla, self).__init__(encoder)
        self.embedding = EmptyReplacement()
        self.model_name = "Vanilla None"