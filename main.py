import argparse
from instructors.RICO_instructor import RICOInstructor
from datasets.RICO_loader import RICOLoader
from utils.metrics import Metrics

from models.BiLSTM_models import EmptyBiLSTM, LinearBiLSTM, EmbeddingBiLSTM
from models.LSTM_models import EmptyLSTM, LinearLSTM, EmbeddingLSTM
from models.GRU_models import EmptyGRU, LinearGRU, EmbeddingGRU
from models.Vanilla_models import EmbeddingVanilla, LinearVanilla, EmptyVanilla
from models.DNN_models import EmptyDNN, LinearDNN, EmbeddingDNN
from models.CNN_models import EmptyCNN, LinearCNN, EmbeddingCNN
from models.Pooling_models import EmptyPooling, LinearPooling, EmbeddingPooling

from utils.utils import create_experiment_folder

def setup_config(parser):
    parser.add_argument("--train_src", default="..\\RicoDatasetLoader\\small_sample\\train", type=str)
    parser.add_argument("--test_src", default="..\\RicoDatasetLoader\\small_sample\\test", type=str)
    parser.add_argument("--experiment_folder", default="experiment0", type=str)
    parser.add_argument("--output_folder", default="experiments", type=str)
    parser.add_argument("--model_folder", default="models", type=str)
    parser.add_argument("--time_epochs", default=True, type=bool)
    parser.add_argument("--time_loading", default=True, type=bool)
    parser.add_argument("--save_models", default=True, type=bool)
    parser.add_argument("--save_metrics", default=True, type=bool)
    parser.add_argument("--metrics_workbook_name", default="metrics_workbook.xlsx", type=str)
    parser.add_argument("--epochs", default=2, type=int)
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = setup_config(parser)
    opt = parser.parse_args()

    create_experiment_folder(opt)

    metrics = Metrics(opt)

    data_loader = RICOLoader(opt)
    dataset = data_loader.load()

    model_list = []

    model_list.append(RICOInstructor(opt, dataset, classifier=EmbeddingDNN(), name="dnn_embedding"))
    model_list.append(RICOInstructor(opt, dataset, classifier=LinearDNN(), name="dnn_linear"))
    model_list.append(RICOInstructor(opt, dataset, classifier=EmptyDNN(), name="dnn_empty"))

    model_list.append(RICOInstructor(opt, dataset, classifier=EmbeddingBiLSTM(), name="bilstm_embedding"))
    model_list.append(RICOInstructor(opt, dataset, classifier=LinearBiLSTM(), name="bilstm_linear"))
    model_list.append(RICOInstructor(opt, dataset, classifier=EmptyBiLSTM(), name="bilstm_empty"))

    model_list.append(RICOInstructor(opt, dataset, classifier=EmbeddingCNN(), name="cnn_embedding"))
    model_list.append(RICOInstructor(opt, dataset, classifier=LinearCNN(), name="cnn_linear"))
    model_list.append(RICOInstructor(opt, dataset, classifier=EmptyCNN(), name="cnn_empty"))

    model_list.append(RICOInstructor(opt, dataset, classifier=EmbeddingLSTM(), name="lstm_embedding"))
    model_list.append(RICOInstructor(opt, dataset, classifier=LinearLSTM(), name="lstm_linear"))
    model_list.append(RICOInstructor(opt, dataset, classifier=EmptyLSTM(), name="lstm_empty"))

    model_list.append(RICOInstructor(opt, dataset, classifier=EmbeddingGRU(), name="gru_embedding"))
    model_list.append(RICOInstructor(opt, dataset, classifier=LinearGRU(), name="gru_linear"))
    model_list.append(RICOInstructor(opt, dataset, classifier=EmptyGRU(), name="gru_empty"))

    model_list.append(RICOInstructor(opt, dataset, classifier=EmbeddingVanilla(), name="vanilla_embedding"))
    model_list.append(RICOInstructor(opt, dataset, classifier=LinearVanilla(), name="vanilla_linear"))
    model_list.append(RICOInstructor(opt, dataset, classifier=EmptyVanilla(), name="vanilla_empty"))

    model_list.append(RICOInstructor(opt, dataset, classifier=EmbeddingPooling(), name="gp_embedding"))
    model_list.append(RICOInstructor(opt, dataset, classifier=LinearPooling(), name="gp_linear"))
    model_list.append(RICOInstructor(opt, dataset, classifier=EmptyPooling(), name="gp_empty"))

    for instructor in model_list:
        try:
            instructor.run(metrics.writer)
        except Exception as e:
            print(e)
            print("Cannot run " + instructor.name)

    metrics.close()