import os

from datasets.IMDB_loader import IMDBLoader
from managers.basic_manager import CoreManager
from utils.IMDB_metric_writer import IMDBMetricWriter
from instructors.IMDB_instructor import IMDBInstructor

from models.BiLSTM_models import EmptyBiLSTM, LinearBiLSTM, EmbeddingBiLSTM
from models.LSTM_models import EmptyLSTM, LinearLSTM, EmbeddingLSTM
from models.GRU_models import EmptyGRU, LinearGRU, EmbeddingGRU
from models.Vanilla_models import EmbeddingVanilla, LinearVanilla, EmptyVanilla
from models.DNN_models import EmptyDNN, LinearDNN, EmbeddingDNN
from models.CNN_models import EmptyCNN, LinearCNN, EmbeddingCNN
from models.Pooling_models import EmptyPooling, LinearPooling, EmbeddingPooling

class IMDBManager(CoreManager):
    def __init__(self, opt):
        super(IMDBManager, self).__init__(opt)
        self.data_writer = IMDBMetricWriter(self.opt)
        self.data_loader = IMDBLoader(self.opt)
        if self.opt.save_models:
            self.opt.IMDB_model_folder = os.path.join(self.opt.experiment_folder, self.opt.IMDB_model_folder)
            os.mkdir(self.opt.IMDB_model_folder)
        
    def setup_instructors(self):
        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=EmbeddingDNN(self.opt.IMDB_vocab_size, self.data_loader.encoder), name="dnn_embedding"))
        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=LinearDNN(self.data_loader.encoder), name="dnn_linear"))
        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=EmptyDNN(self.data_loader.encoder), name="dnn_empty"))

        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=EmbeddingBiLSTM(self.opt.IMDB_vocab_size, self.data_loader.encoder), name="bilstm_embedding"))
        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=LinearBiLSTM(self.data_loader.encoder), name="bilstm_linear"))
        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=EmptyBiLSTM(self.data_loader.encoder), name="bilstm_empty"))

        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=EmbeddingCNN(self.opt.IMDB_vocab_size, self.data_loader.masked_encoder), name="cnn_embedding"))
        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=LinearCNN(self.data_loader.masked_encoder), name="cnn_linear"))
        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=EmptyCNN(self.data_loader.masked_encoder), name="cnn_empty"))

        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=EmbeddingLSTM(self.opt.IMDB_vocab_size, self.data_loader.encoder), name="lstm_embedding"))
        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=LinearLSTM(self.data_loader.encoder), name="lstm_linear"))
        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=EmptyLSTM(self.data_loader.encoder), name="lstm_empty"))

        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=EmbeddingGRU(self.opt.IMDB_vocab_size, self.data_loader.encoder), name="gru_embedding"))
        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=LinearGRU(self.data_loader.encoder), name="gru_linear"))
        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=EmptyGRU(self.data_loader.encoder), name="gru_empty"))

        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=EmbeddingVanilla(self.opt.IMDB_vocab_size, self.data_loader.encoder), name="vanilla_embedding"))
        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=LinearVanilla(self.data_loader.encoder), name="vanilla_linear"))
        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=EmptyVanilla(self.data_loader.encoder), name="vanilla_empty"))

        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=EmbeddingPooling(self.opt.IMDB_vocab_size, self.data_loader.encoder), name="gp_embedding"))
        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=LinearPooling(self.data_loader.encoder), name="gp_linear"))
        self.model_list.append(IMDBInstructor(self.opt, self.dataset, classifier=EmptyPooling(self.data_loader.encoder), name="gp_empty"))