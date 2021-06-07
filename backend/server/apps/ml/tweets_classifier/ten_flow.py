import tensorflow as tf
from tensorflow import keras
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

class tens_flow_classifier:
    def __init__(self):
        path_to_artifacts = "../../research"
        maxlen = 50
        self.loaded_model = tf.keras.models.load_model(path_to_artifacts + "model/my_model")

    def get_pickle(self, pick):
        # loading
        with open(self.path_to_artifacts + 'tokenizer.pickle', 'rb') as handle:
            p = pickle.load(handle)
        return p

    def get_sequences(self, token_maker, input_data):
        seq = token_maker.texts_to_sequences(input_data)
        padded = pad_sequences(seq, maxlen=self.maxlen, truncating='post',
                               padding='post')
        return padded

    def preprocessing(self, input_data):
        token_maker = self.get_pickle('tokenizer.pickle')
        input_data_seq = self.get_sequences(self, token_maker, input_data)
        return input_data_seq

    def predict(self, input_data):
        return self.loaded_model.predict(input_data)

    def postprocessing(self, input_data):
        label_classe = self.get_pickle('label_class.pickle')
        pred_class = label_classe.index_to_class[np.argmax(input_data).astype('uint8')]
        return pred_class


    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction


