import tensorflow as tf
from instructors.instructor import Instructor
from models.classifier_loss import discriminator_loss
import time
import pandas as pd

class RICOInstructor(Instructor):
    def __init__(self, opt, data, classifier = None, name=""):
        super(RICOInstructor, self).__init__(opt, data, classifier=classifier, name=name)
        self.all_training_losses = []

    def run(self, writer=None):
        self.train(writer)
        self.test()
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
            self.save_losses(writer)

    def save_losses(self, writer):
        dataframe_losses = pd.DataFrame(data=self.all_training_losses)
        dataframe_losses.to_excel(writer, sheet_name=self.classifier.model_name)