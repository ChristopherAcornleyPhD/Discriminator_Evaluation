from datasets.dataset_loader import DatasetLoader
from layers.max_layer import MaxLayer
import os
import time
import random
import numpy as np
import tensorflow as tf

class RICOLoader(DatasetLoader):
    def __init__(self, options):
        super(RICOLoader, self).__init__()
        self.opt = options
        self.fake_generator = MaxLayer()
        self.lowest_lines = 10000000
        self.highest_lines = 0

    def load_train_data_from_folder(self, target_folder):
        file_data = []
        list_of_files = os.listdir(target_folder)
        for file_name in list_of_files:
            file_name = target_folder+'\\'+file_name
            file = open(file_name, 'r')
            fd = []
            file_length = len(file.readlines())
            if file_length > self.highest_lines:
                self.highest_lines = file_length
            if file_length < self.lowest_lines:
                self.lowest_lines = file_length
            for line in file.readlines():
                line_list = line.split(' ')
                if (len(line_list) != 6) or (line_list[0] == '#'):
                    continue
                for item in line_list:
                    item = item.strip()
                    if item.isdigit():
                        item = float(item)
                        if item > 100.0:
                            item = 100.0
                        fd.append(item)
            file_data.append(fd)
            file.close()
        return np.array(file_data, dtype=object)
    
    def generate_sample_data_batch(self, file_data, num_expands = 1):
        disc_data = []
        gen_data = []
        for batch in file_data:
            batch_disc, batch_gen = self.generate_sample_data(batch, num_expands=num_expands)
            disc_data.append(batch_disc)
            gen_data.append(batch_gen)
        return disc_data, gen_data
    
    def generate_sample_data(self, file_data, num_expands = 1):
        disc_data = []
        gen_data = []

        for item_iter in range(len(file_data)):
            p_samples = file_data[item_iter]
            tensor = tf.convert_to_tensor(p_samples)
            for _ in range(num_expands):
                tensor = tf.expand_dims(tensor, axis=0)
            disc_data.append(tensor)
            random_gen = np.random.uniform(0.0, 1.0, (1, len(file_data[item_iter]), 101))
            pred = self.fake_generator(random_gen)
            gen_data.append(pred)

        return disc_data, gen_data
    
    def combine_testing_data(self, true_test, gen_test):
        gen_length = len(gen_test)
        true_length = len(true_test)
        
        if gen_length != true_length:
            raise Exception("Generated and True samples do not match in size.")

        data = true_test + gen_test

        true_label = [1] * true_length
        false_label = [0] * gen_length

        labels = true_label + false_label

        c = list(zip(data, labels))
        random.shuffle(c)

        data_shuffled, labels_shuffled = zip(*c)

        return {"data" : data_shuffled, "labels" : labels_shuffled}

    
    def load(self):
        print("Loading RICO dataset...")
        start_time = time.perf_counter()
        if os.path.exists(self.opt.RICO_train_src) and os.path.exists(self.opt.RICO_test_src):
            raw_file_data = self.load_train_data_from_folder(self.opt.RICO_train_src)
            raw_tester_data = self.load_train_data_from_folder(self.opt.RICO_test_src)

            files_per_batch = 100
            batch_size = int(len(raw_file_data)  / files_per_batch)
            file_data = np.array_split(raw_file_data, batch_size)

            disc_data_train, gen_data_train = self.generate_sample_data_batch(file_data)
            disc_data_test, gen_data_test = self.generate_sample_data(raw_tester_data)
            self.train_data = {"disc": disc_data_train, "gen": gen_data_train}
            self.test_data = self.combine_testing_data(disc_data_test, gen_data_test)
            self.batch_size = batch_size

        else:
            print("Source folders do not exist.")

        end_time = time.perf_counter()

        if self.opt.time_loading:
            print("Time taken to load RICO Dataset: {}".format(end_time - start_time))
        
        return self