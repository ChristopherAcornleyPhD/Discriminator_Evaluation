import tensorflow as tf
from models.classifiers import DenseClassifier
from layers.linear_replacement import LinearReplacement
from layers.empty_replacement import EmptyReplacement

class EmbeddingDNN(DenseClassifier):
    def __init__(self, vocal_size, encoder=None):
        super(EmbeddingDNN, self).__init__(encoder)
        self.embedding = tf.keras.layers.Embedding(vocal_size, 64)
        self.model_name = "DNN Embedding"

class LinearDNN(DenseClassifier):
    def __init__(self, encoder=None):
        super(LinearDNN, self).__init__(encoder)
        self.embedding = LinearReplacement(64)
        self.model_name = "DNN Linear"

class EmptyDNN(DenseClassifier):
    def __init__(self, encoder=None):
        super(EmptyDNN, self).__init__(encoder)
        self.embedding = EmptyReplacement()
        self.model_name = "DNN None"