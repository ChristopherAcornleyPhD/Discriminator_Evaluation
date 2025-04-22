import tensorflow as tf
from classifiers import ConvClassifier
from layers.LinearReplacement import LinearReplacement
from layers.EmptyReplacement import EmptyReplacement

class EmbeddingCNN(ConvClassifier):
    def __init__(self):
        super(EmbeddingCNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(101, 64)

class LinearCNN(ConvClassifier):
    def __init__(self):
        super(LinearCNN, self).__init__()
        self.embedding = LinearReplacement(64)

class EmptyCNN(ConvClassifier):
    def __init__(self):
        super(EmptyCNN, self).__init__()
        self.embedding = EmptyReplacement()