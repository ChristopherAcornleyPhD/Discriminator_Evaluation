class CoreManager:
    def __init__(self, opt):
        self.opt = opt
        self.model_list = []
        self.data_writer = None
        self.data_loader = None

    def load_data(self):
        self.dataset = self.data_loader.load()

    def setup_instructors(self):
        pass

    def run_instructors(self):
        for instructor in self.model_list:
            #try:
            instructor.run(self.data_writer)
            # except Exception as e:
            #     print(e)
            #     print("Cannot run " + instructor.name)
        self.data_writer.close()

    def run(self):
        self.load_data()
        self.setup_instructors()
        self.run_instructors()