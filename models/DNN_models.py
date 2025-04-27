import tensorflow as tf
from models.classifiers import DenseClassifier
from layers.linear_replacement import LinearReplacement
from layers.empty_replacement import EmptyReplacement

class EmbeddingDNN(DenseClassifier):
    def __init__(self):
        super(EmbeddingDNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(101, 64)
        self.model_name = "DNN Embedding"

class LinearDNN(DenseClassifier):
    def __init__(self):
        super(LinearDNN, self).__init__()
        self.embedding = LinearReplacement(64)
        self.model_name = "DNN Linear"

class EmptyDNN(DenseClassifier):
    def __init__(self):
        super(EmptyDNN, self).__init__()
        self.embedding = EmptyReplacement()
        self.model_name = "DNN None"