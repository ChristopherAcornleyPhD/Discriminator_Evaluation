import tensorflow as tf

class LossCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LossCallback, self).__init__()
        self.all_batch_losses = []
    def on_train_batch_end(self, batch, logs=None):
        self.all_batch_losses.append(logs.get('loss'))