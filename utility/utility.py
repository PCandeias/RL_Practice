from pathlib import Path
from keras.models import clone_model

def file_exists(filename):
    file = Path(filename)
    return file.is_file()

def copy_model(model):
    new_model = clone_model(model)
    new_model.set_weights(model.get_weights())
    return new_model
