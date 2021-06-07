import tensorflow as tf
from tensorflow import keras
import pandas as pd

class tens_flow_classifier:
    def __init__(self):
        path_to_artifacts = "../../research"
        self.loaded_model = tf.keras.models.load_model(path_to_artifacts + "model/my_model")


    def preprocessing(self, input_data):
        input_data_seq = get_sequences(tokenizer, input_data)
