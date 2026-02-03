import os
import sys
import keras
import pickle

from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from keras.utils import pad_sequences

from hate.components.data_transforamation import DataTransformation
from hate.entity.config_entity import (
    DataTransformationConfig,
    ModelTrainerConfig
)


class PredictionPipeline:
    def __init__(self):
        try:
            self.model_trainer_config = ModelTrainerConfig()

            self.model_path = self.model_trainer_config.TRAINED_MODEL_PATH
            self.tokenizer_path = os.path.join(
                os.path.dirname(self.model_path),
                "tokenizer.pickle"
            )

            self.data_transformation = DataTransformation(
                data_transformation_config=DataTransformationConfig(),
                data_ingestion_artifacts=None
            )

            logging.info(f"PredictionPipeline initialized")
            logging.info(f"Model path: {self.model_path}")
            logging.info(f"Tokenizer path: {self.tokenizer_path}")

        except Exception as e:
            raise CustomException(e, sys)


    def get_local_model(self) -> str:
        try:
            if not os.path.isfile(self.model_path):
                raise Exception(
                    f"Trained model not found at path: {self.model_path}"
                )

            logging.info("Trained model found")
            return self.model_path

        except Exception as e:
            raise CustomException(e, sys)

  
    def predict(self, text):
        logging.info("Running prediction")
        try:
            model_path = self.get_local_model()
            model = keras.models.load_model(model_path)

            if not os.path.isfile(self.tokenizer_path):
                raise Exception("Tokenizer file not found")

            with open(self.tokenizer_path, "rb") as handle:
                tokenizer = pickle.load(handle)

            cleaned_text = self.data_transformation.concat_data_cleaning(text)
            cleaned_text = [cleaned_text]

            seq = tokenizer.texts_to_sequences(cleaned_text)
            padded = pad_sequences(seq, maxlen=MAX_LEN)

            pred = model.predict(padded)

            if pred[0][0] >= 0.35:
                return "hate and abusive"
            else:
                return "no hate"

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self, text):
        try:
            return self.predict(text)
        except Exception as e:
            raise CustomException(e, sys)
