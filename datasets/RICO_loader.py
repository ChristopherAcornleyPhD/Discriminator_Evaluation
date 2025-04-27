from datasets.dataset_loader import DatasetLoader
from layers.max_layer import MaxLayer
import os
import time
import numpy as np
import tensorflow as tf

class RICOLoader(DatasetLoader):
    def __init__(self, options):
        super(RICOLoader, self).__init__()
        self.opt = options
        self.fake_generator = MaxLayer()

    def load_train_data_from_folder(self, target_folder):
        file_data = []
        list_of_files = os.listdir(target_folder)
        for file_name in list_of_files:
            file_name = target_folder+'\\'+file_name
            file = open(file_name, 'r')
            fd = []
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
    
    def generate_sample_data(self, file_data, num_expands = 1):
        disc_data = []
        gen_data = []

        for batch_iter in range(len(file_data)):
            disc_batch = []
            gen_batch = []
            for item_iter in range(len(file_data[batch_iter])):
                p_samples = file_data[batch_iter][item_iter]
                tensor = tf.convert_to_tensor(p_samples)
                for _ in range(num_expands):
                    tensor = tf.expand_dims(tensor, axis=0)
                disc_batch.append(tensor)
                random_gen = np.random.uniform(0.0, 1.0, (1, len(file_data[batch_iter][item_iter]), 101))
                pred = self.fake_generator(random_gen)
                gen_batch.append(pred)
            disc_data.append(disc_batch)  
            gen_data.append(gen_batch)

        return disc_data, gen_data
    
    def load(self):
        start_time = time.perf_counter()
        if os.path.exists(self.opt.train_src) and os.path.exists(self.opt.test_src):
            raw_file_data = self.load_train_data_from_folder(self.opt.train_src)
            raw_tester_data = self.load_train_data_from_folder(self.opt.test_src)

            files_per_batch = 100
            batch_size = int(len(raw_file_data)  / files_per_batch)
            file_data = np.array_split(raw_file_data, batch_size)
            tester_data = np.array_split(raw_tester_data, batch_size)

            disc_data_train, gen_data_train = self.generate_sample_data(file_data)
            disc_data_test, gen_data_test = self.generate_sample_data(tester_data)
            self.train_data = {"disc": disc_data_train, "gen": gen_data_train}
            self.test_data = {"disc": disc_data_test, "gen": gen_data_test}
            self.batch_size = batch_size

        else:
            print("Source folders do not exist.")

        end_time = time.perf_counter()

        if self.opt.time_loading:
            print("Time taken to load RICO Dataset: {}".format(end_time - start_time))
        
        return self