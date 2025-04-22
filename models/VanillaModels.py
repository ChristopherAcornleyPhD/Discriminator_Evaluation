import tensorflow as tf
from classifiers import RecurrentClassifier
from layers.LinearReplacement import LinearReplacement
from layers.EmptyReplacement import EmptyReplacement

class VanillaClassifier(RecurrentClassifier):
    def __init__(self):
        super(VanillaClassifier, self).__init__()
        self.recurrent = tf.keras.layers.SimpleRNN(64)

class EmbeddingVanilla(VanillaClassifier):
    def __init__(self):
        super(EmbeddingVanilla, self).__init__()
        self.embedding = tf.keras.layers.Embedding(101, 64)

class LinearVanilla(VanillaClassifier):
    def __init__(self):
        super(LinearVanilla, self).__init__()
        self.embedding = LinearReplacement(64)

class EmptyVanilla(VanillaClassifier):
    def __init__(self):
        super(EmptyVanilla, self).__init__()
        self.embedding = EmptyReplacement()