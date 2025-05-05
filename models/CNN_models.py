import tensorflow as tf
from models.classifiers import ConvClassifier
from layers.linear_replacement import LinearReplacement
from layers.empty_replacement import EmptyReplacement

class EmbeddingCNN(ConvClassifier):
    def __init__(self, vocal_size, encoder=None):
        super(EmbeddingCNN, self).__init__(encoder)
        self.embedding = tf.keras.layers.Embedding(vocal_size, 64, mask_zero=True)
        self.model_name = "CNN Embedding"

class LinearCNN(ConvClassifier):
    def __init__(self, encoder=None):
        super(LinearCNN, self).__init__(encoder)
        self.embedding = LinearReplacement(64)
        self.model_name = "CNN Linear"

class EmptyCNN(ConvClassifier):
    def __init__(self, encoder=None):
        super(EmptyCNN, self).__init__(encoder)
        self.embedding = EmptyReplacement()
        self.model_name = "CNN None"