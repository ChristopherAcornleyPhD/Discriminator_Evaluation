class Instructor:
    def __init__(self):
        self.name = ""
        self.classifier = None
        self.train_data = None
        self.test_data = None
    
    def run(self):
        print("Running " + self.name + " instructor.")
        pass

    def train(self):
        
