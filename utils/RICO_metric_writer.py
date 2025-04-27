from utils.metric_writer import MetricWriter
import pandas as pd

class RICOMetricWriter(MetricWriter):
    def __init__(self, opt):
        super(RICOMetricWriter, self).__init__(opt)
        start_data = {"Metric": ["Accuracy", "Recall", "Precision", "F1 Score"]}
        self.data_to_save = {"losses": pd.DataFrame(), "metrics": pd.DataFrame.from_dict(start_data)}

    def save_losses(self, model_name, data):
        dataframe = self.data_to_save["losses"]
        dataframe[model_name] = data

    def save_metrics(self, model_name, data):
        dataframe = self.data_to_save["metrics"]
        dataframe[model_name] = data