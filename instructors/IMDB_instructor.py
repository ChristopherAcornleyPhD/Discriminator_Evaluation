import tensorflow as tf
from instructors.instructor import Instructor
from utils.utils import calculate_F1Score
from utils.loss_callback import LossCallback
import time
import os

class IMDBInstructor(Instructor):
    def __init__(self, opt, data, classifier = None, name=""):
        super(IMDBInstructor, self).__init__(opt, data, classifier=classifier, name=name)
        self.all_training_losses = []

    def run(self, writer=None):
        if self.opt.IMDB_training:
            self.train(writer)
        if self.opt.IMDB_testing:
            self.test(writer)
        if self.opt.save_models and self.opt.IMDB_training:
            self.save_model()

    def train(self, writer):
        print("Started training of {}".format(self.classifier.model_name))

        loss_callback = LossCallback()
        train_timer_start = time.perf_counter()
        train_dataset = self.data.train_dataset
        self.classifier.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0), tf.keras.metrics.Recall(thresholds=0), tf.keras.metrics.Precision(thresholds=0)])
        self.classifier.fit(train_dataset, epochs=self.opt.epochs, callbacks = [loss_callback])
        train_timer_end = time.perf_counter()

        if self.opt.save_metrics or writer is not None:
            writer.save_losses(self.classifier.model_name, loss_callback.all_batch_losses)

        if self.opt.time_epochs:
            print('Finished training IMDB {}. Time Taken: {:.0f} seconds'.format(self.classifier.model_name, train_timer_end - train_timer_start))

    def test(self, writer):
        _, test_acc, test_recall, test_precision = self.classifier.evaluate(self.data.test_dataset)

        # f1
        f1_metric = calculate_F1Score(test_precision, test_recall)

        print("Model Name: {}\nAccuracy: {}\nRecall: {}\nPrecision: {}\nF1 Score: {}".format(self.classifier.model_name, test_acc, test_recall, test_precision, f1_metric))

        if self.opt.save_metrics or writer is not None:
            writer.save_metrics(self.classifier.model_name, [test_acc, test_recall, test_precision, f1_metric])

    def save_model(self):
        pass
        #self.classifier.save_weights(os.path.join(self.opt.IMDB_model_folder, self.classifier.model_name))