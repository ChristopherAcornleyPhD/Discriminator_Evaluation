import os
import pandas as pd

class MetricWriter():
    def __init__(self, opt):
        self.opt = opt
        self.data_to_save = dict()
        self.workbook_name = ""

    def prepare_workbook(self):
        if self.opt.save_metrics:
            try:
                self.workbook_path = os.path.join(self.opt.experiment_folder, self.workbook_name)
                self.writer = pd.ExcelWriter(self.workbook_path)
            except Exception as e:
                print(e)
                print("Cannot create workbook")
                self.writer = None
                self.workbook_path = ""

    def close(self):
        if self.writer is not None:
            for key, value in self.data_to_save.items():
                value.to_excel(self.writer, sheet_name=key)
            self.writer.close()