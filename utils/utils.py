import os
from keras import backend as K

def create_experiment_folder(opt):
    if not opt.save_models and not opt.save_metrics:
        return

    try:
        if not os.path.exists(opt.output_folder):
            os.mkdir(opt.output_folder)
        
        index = 0
        experiment_folder = opt.experiment_folder

        while True: 
            experiment_folder_path = os.path.join(opt.output_folder, experiment_folder)

            if os.path.exists(experiment_folder_path):
                index += 1
                experiment_folder = "experiment" + str(index)
            else:
                os.mkdir(experiment_folder_path)
                opt.experiment_folder = experiment_folder_path
                return
            
    except:
        print("Cannot create experiment folder")
        opt.save_models = False
        opt.save_metrics = False

def calculate_F1Score(precision, recall):
    return 2*((precision*recall)/(precision+recall+K.epsilon()))