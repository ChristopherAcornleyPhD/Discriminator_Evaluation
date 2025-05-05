import tensorflow as tf

class Instructor:
    def __init__(self, opt, data, classifier = None, name = ""):
        self.name = name
        self.opt = opt
        self.data = data
        self.classifier = classifier
        self.optimizer = tf.keras.optimizers.Adam()

    def train(self):
        print("Training")
        pass

    def test(self):
        print("Testing")
        pass

    def save_model(self):
        print("Saving Model")
        pass