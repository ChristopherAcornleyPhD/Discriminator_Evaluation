import tensorflow as tf
from classifiers import ConvClassifier
from layers.linear_replacement import LinearReplacement
from layers.empty_replacement import EmptyReplacement

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