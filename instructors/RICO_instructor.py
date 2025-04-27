import tensorflow as tf
from instructors.instructor import Instructor
from models.classifier_loss import discriminator_loss
from utils.utils import calculate_F1Score
import time
import pandas as pd

class RICOInstructor(Instructor):
    def __init__(self, opt, data, classifier = None, name=""):
        super(RICOInstructor, self).__init__(opt, data, classifier=classifier, name=name)
        self.all_training_losses = []

    def run(self, writer=None):
        if self.opt.training:
            self.train(writer)
        if self.opt.testing:
            self.test(writer)
        if self.opt.save_models:
            self.save_model()

    def train(self, writer):
        disc_data = self.data.train_data["disc"]
        gen_data = self.data.train_data["gen"]
        for epoch in range(self.opt.epochs):
            timer_start = time.perf_counter()
            total_loss = 0
            for batch in range(self.data.batch_size):
                with tf.GradientTape() as disc_tape:
                    predictions_true = []
                    for item in disc_data[batch]:
                        predictions_true.append(self.classifier(item, training=True))

                    predictions_false = []
                    for pred in gen_data[batch]:
                        predictions_false.append(self.classifier(pred, training=True))
                    
                    disc_loss = discriminator_loss(predictions_true, predictions_false)
        
                    if self.opt.save_metrics:
                        self.all_training_losses.append(disc_loss.numpy())
                    total_loss += disc_loss

                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.classifier.variables)
                self.optimizer.apply_gradients(zip(gradients_of_discriminator, self.classifier.variables))
            print('Average loss for epoch: {}'.format(total_loss/float(self.data.batch_size)))
            timer_end = time.perf_counter()
            if self.opt.time_epochs:
                print('Finished epoch {}. Time Taken: {:.0f} seconds'.format(epoch + 1, timer_end - timer_start))

        if self.opt.save_metrics or writer is not None:
            writer.save_losses(self.classifier.model_name, self.all_training_losses)


    def test(self, writer):
        if not self.opt.training:
            # Check if the folder to load models from is valid
            # load model weights
            # otherwise skip
            pass

        # # test model
        predictions = []
        testing_data = self.data.test_data["data"]
        labels = self.data.test_data["labels"]
        for item in testing_data:
            predictions.append(self.classifier(item))

        prediction_tensor = tf.concat(predictions,axis=1)
        label_tensor = tf.expand_dims(tf.convert_to_tensor(labels), axis = 0)

        # accuracy - TP
        Accuracy = tf.keras.metrics.binary_accuracy(labels, prediction_tensor)
        acc_metric = Accuracy.numpy()[0]

        # recall
        recall = tf.keras.metrics.Recall()
        recall.update_state(label_tensor, prediction_tensor)
        recall_metric = recall.result().numpy()

        # precision
        precision = tf.keras.metrics.Precision()
        precision.update_state(label_tensor, prediction_tensor)
        precision_metric = precision.result().numpy()

        # f1
        f1_metric = calculate_F1Score(precision_metric, recall_metric)

        print("Model Name: {}\nAccuracy: {}\nRecall: {}\nPrecision: {}\nF1 Score: {}".format(self.classifier.model_name, acc_metric, recall_metric, precision_metric, f1_metric))

        if self.opt.save_metrics or writer is not None:
            writer.save_metrics(self.classifier.model_name, [acc_metric, recall_metric, precision_metric, f1_metric])