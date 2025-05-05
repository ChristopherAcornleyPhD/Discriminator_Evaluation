import tensorflow as tf
from models.classifiers import PoolingClassifier
from layers.linear_replacement import LinearReplacement
from layers.empty_replacement import EmptyReplacement

class EmbeddingPooling(PoolingClassifier):
    def __init__(self, vocal_size, encoder=None):
        super(EmbeddingPooling, self).__init__(encoder)
        self.embedding = tf.keras.layers.Embedding(vocal_size, 64)
        self.model_name = "GP Embedding"

class LinearPooling(PoolingClassifier):
    def __init__(self, encoder=None):
        super(LinearPooling, self).__init__(encoder)
        self.embedding = LinearReplacement(64)
        self.model_name = "GP Linear"

class EmptyPooling(PoolingClassifier):
    def __init__(self, encoder=None):
        super(EmptyPooling, self).__init__(encoder)
        self.embedding = EmptyReplacement()
        self.model_name = "GP None"