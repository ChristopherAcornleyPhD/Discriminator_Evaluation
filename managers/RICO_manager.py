import os

from datasets.RICO_loader import RICOLoader
from managers.basic_manager import CoreManager
from utils.RICO_metric_writer import RICOMetricWriter
from instructors.RICO_instructor import RICOInstructor

from models.BiLSTM_models import EmptyBiLSTM, LinearBiLSTM, EmbeddingBiLSTM
from models.LSTM_models import EmptyLSTM, LinearLSTM, EmbeddingLSTM
from models.GRU_models import EmptyGRU, LinearGRU, EmbeddingGRU
from models.Vanilla_models import EmbeddingVanilla, LinearVanilla, EmptyVanilla
from models.DNN_models import EmptyDNN, LinearDNN, EmbeddingDNN
from models.CNN_models import EmptyCNN, LinearCNN, EmbeddingCNN
from models.Pooling_models import EmptyPooling, LinearPooling, EmbeddingPooling

class RICOManager(CoreManager):
    def __init__(self, opt):
        super(RICOManager, self).__init__(opt)
        self.data_writer = RICOMetricWriter(self.opt)
        self.data_loader = RICOLoader(self.opt)
        if self.opt.save_models:
            self.opt.RICO_model_folder = os.path.join(self.opt.experiment_folder, self.opt.RICO_model_folder)
            os.mkdir(self.opt.RICO_model_folder)   

    def setup_instructors(self):
        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=EmbeddingDNN(self.opt.RICO_vocab_size), name="dnn_embedding"))
        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=LinearDNN(), name="dnn_linear"))
        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=EmptyDNN(), name="dnn_empty"))

        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=EmbeddingBiLSTM(self.opt.RICO_vocab_size), name="bilstm_embedding"))
        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=LinearBiLSTM(), name="bilstm_linear"))
        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=EmptyBiLSTM(), name="bilstm_empty"))

        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=EmbeddingCNN(self.opt.RICO_vocab_size), name="cnn_embedding"))
        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=LinearCNN(), name="cnn_linear"))
        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=EmptyCNN(), name="cnn_empty"))

        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=EmbeddingLSTM(self.opt.RICO_vocab_size), name="lstm_embedding"))
        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=LinearLSTM(), name="lstm_linear"))
        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=EmptyLSTM(), name="lstm_empty"))

        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=EmbeddingGRU(self.opt.RICO_vocab_size), name="gru_embedding"))
        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=LinearGRU(), name="gru_linear"))
        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=EmptyGRU(), name="gru_empty"))

        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=EmbeddingVanilla(self.opt.RICO_vocab_size), name="vanilla_embedding"))
        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=LinearVanilla(), name="vanilla_linear"))
        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=EmptyVanilla(), name="vanilla_empty"))

        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=EmbeddingPooling(self.opt.RICO_vocab_size), name="gp_embedding"))
        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=LinearPooling(), name="gp_linear"))
        self.model_list.append(RICOInstructor(self.opt, self.dataset, classifier=EmptyPooling(), name="gp_empty"))