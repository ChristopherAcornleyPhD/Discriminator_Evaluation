import argparse

from managers.RICO_manager import RICOManager
from managers.IMDB_manager import IMDBManager

from utils.utils import create_experiment_folder

def setup_config(parser):
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--experiment_folder", default="experiment0", type=str)
    parser.add_argument("--output_folder", default="experiments", type=str)
    parser.add_argument("--time_epochs", default=True, type=bool)
    parser.add_argument("--time_loading", default=True, type=bool)
    parser.add_argument("--save_models", default=True, type=bool)
    parser.add_argument("--save_metrics", default=True, type=bool)

    parser.add_argument("--RICO_training", default=True, type=bool)
    parser.add_argument("--RICO_testing", default=True, type=bool)
    parser.add_argument("--RICO_model_folder", default="RICO_models", type=str)
    parser.add_argument("--RICO_vocab_size", default=101, type=int)
    parser.add_argument("--RICO_train_src", default="..\\RicoDatasetLoader\\small_sample\\train", type=str)
    parser.add_argument("--RICO_test_src", default="..\\RicoDatasetLoader\\small_sample\\test", type=str)
    parser.add_argument("--RICO_metrics_workbook_name", default="rico_metrics_workbook.xlsx", type=str)

    parser.add_argument("--IMDB_training", default=True, type=bool)
    parser.add_argument("--IMDB_testing", default=True, type=bool)
    parser.add_argument("--IMDB_model_folder", default="IMDB_models", type=str)
    parser.add_argument("--IMDB_buffer_size", default=10000, type=int)
    parser.add_argument("--IMDB_batch_size", default=64, type=int)
    parser.add_argument("--IMDB_vocab_size", default=1000, type=int)
    parser.add_argument("--IMDB_metrics_workbook_name", default="imdb_metrics_workbook.xlsx", type=str)
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = setup_config(parser)
    opt = parser.parse_args()

    create_experiment_folder(opt)

    if opt.RICO_training or opt.RICO_testing:
        rico_manager = RICOManager(opt)
        rico_manager.run()

    if opt.IMDB_training or opt.IMDB_testing:
        imdb_manager = IMDBManager(opt)
        imdb_manager.run()