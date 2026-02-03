import os
import sys
import keras
import pickle

from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from keras.utils import pad_sequences

from hate.components.data_transforamation import DataTransformation
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts


class PredictionPipeline:
    def __init__(self):
        try:
            # Local model paths
            self.model_path = os.path.join(
                "artifacts",
                "trained_model",
                "model.h5"
            )

            self.tokenizer_path = os.path.join(
                "artifacts",
                "trained_model",
                "tokenizer.pickle"
            )

            # Only for text cleaning reuse
            self.data_transformation = DataTransformation(
                data_transformation_config=DataTransformationConfig(),
                data_ingestion_artifacts=None
            )

            logging.info("PredictionPipeline initialized (local mode)")

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------
    # LOAD MODEL LOCALLY
    # -------------------------------------------------
    def get_local_model(self) -> str:
        """
        Method Name :   get_local_model
        Description :   Load trained model from local artifacts
        Output      :   model_path
        """
        try:
            if not os.path.isfile(self.model_path):
                raise Exception("Trained model not found at local path")

            logging.info("Local trained model found")
            return self.model_path

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------
    # PREDICT
    # -------------------------------------------------
    def predict(self, text):
        logging.info("Running the predict function (local model)")
        try:
            # Load model
            model_path = self.get_local_model()
            model = keras.models.load_model(model_path)

            # Load tokenizer
            with open(self.tokenizer_path, "rb") as handle:
                tokenizer = pickle.load(handle)

            # Clean text using same transformation logic
            cleaned_text = self.data_transformation.concat_data_cleaning(text)
            cleaned_text = [cleaned_text]

            # Tokenize & pad
            seq = tokenizer.texts_to_sequences(cleaned_text)
            padded = pad_sequences(seq, maxlen=MAX_LEN)

            # Predict
            pred = model.predict(padded)

            if pred[0][0] > 0.5:
                return "hate and abusive"
            else:
                return "no hate"

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------
    # RUN PIPELINE
    # -------------------------------------------------
    def run_pipeline(self, text):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            predicted_text = self.predict(text)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_text

        except Exception as e:
            raise CustomException(e, sys)
