import tensorflow as tf
from models.classifiers import PoolingClassifier
from layers.linear_replacement import LinearReplacement
from layers.empty_replacement import EmptyReplacement

class EmbeddingPooling(PoolingClassifier):
    def __init__(self):
        super(EmbeddingPooling, self).__init__()
        self.embedding = tf.keras.layers.Embedding(101, 64)
        self.model_name = "GP Embedding"

class LinearPooling(PoolingClassifier):
    def __init__(self):
        super(LinearPooling, self).__init__()
        self.embedding = LinearReplacement(64)
        self.model_name = "GP Linear"

class EmptyPooling(PoolingClassifier):
    def __init__(self):
        super(EmptyPooling, self).__init__()
        self.embedding = EmptyReplacement()
        self.model_name = "GP None"