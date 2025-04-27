import tensorflow as tf
from models.classifiers import RecurrentClassifier
from layers.linear_replacement import LinearReplacement
from layers.empty_replacement import EmptyReplacement

class VanillaClassifier(RecurrentClassifier):
    def __init__(self):
        super(VanillaClassifier, self).__init__()
        self.recurrent = tf.keras.layers.SimpleRNN(64)

class EmbeddingVanilla(VanillaClassifier):
    def __init__(self):
        super(EmbeddingVanilla, self).__init__()
        self.embedding = tf.keras.layers.Embedding(101, 64)
        self.model_name = "Vanilla Embedding"

class LinearVanilla(VanillaClassifier):
    def __init__(self):
        super(LinearVanilla, self).__init__()
        self.embedding = LinearReplacement(64)
        self.model_name = "Vanilla Linear"

class EmptyVanilla(VanillaClassifier):
    def __init__(self):
        super(EmptyVanilla, self).__init__()
        self.embedding = EmptyReplacement()
        self.model_name = "Vanilla None"