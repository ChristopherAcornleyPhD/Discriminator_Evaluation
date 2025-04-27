import os
import pandas as pd

class Metrics():
    def __init__(self, opt):
        self.opt = opt
        if self.opt.save_metrics:
            try:
                self.workbook_path = os.path.join(self.opt.experiment_folder, self.opt.metrics_workbook_name)
                self.writer = pd.ExcelWriter(self.workbook_path)
            except Exception as e:
                print(e)
                print("Cannot create workbook")
                self.writer = None
                self.workbook_path = ""

    def close(self):
        if self.writer is not None:
            self.writer.close()