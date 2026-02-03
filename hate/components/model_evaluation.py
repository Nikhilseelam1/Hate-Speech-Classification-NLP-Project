import os
import sys
import pickle
import keras
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

from hate.logger import logging
from hate.exception import CustomException
from hate.constants import *
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import (
    ModelEvaluationArtifacts,
    ModelTrainerArtifacts,
    DataTransformationArtifacts,
)
from keras.utils import pad_sequences


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifacts: ModelTrainerArtifacts,
        data_transformation_artifacts: DataTransformationArtifacts,
    ):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts

    def evaluate(self):
        try:
            logging.info("Starting model evaluation")

            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path)

            x_test = (
                x_test[TWEET]
                .fillna("")
                .astype(str)
                .str.strip()
            )
            y_test = y_test[LABEL].squeeze()

            tokenizer_path = os.path.join(
                os.path.dirname(self.model_trainer_artifacts.trained_model_path),
                "tokenizer.pickle",
            )

            with open(tokenizer_path, "rb") as handle:
                tokenizer = pickle.load(handle)

            model = keras.models.load_model(
                self.model_trainer_artifacts.trained_model_path
            )

            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(
                test_sequences,
                maxlen=MAX_LEN,
            )

            loss, accuracy = model.evaluate(
                test_sequences_matrix,
                y_test,
                verbose=0,
            )

            logging.info(f"Test Accuracy: {accuracy}")

            predictions = model.predict(test_sequences_matrix)
            y_pred = (predictions >= 0.5).astype(int).ravel()

            cm = confusion_matrix(y_test, y_pred)
            logging.info(f"Confusion Matrix:\n{cm}")

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        try:
            logging.info("Initiating model evaluation")

            accuracy = self.evaluate()

            is_model_accepted = True

            model_evaluation_artifacts = ModelEvaluationArtifacts(
                is_model_accepted=is_model_accepted
            )

            logging.info("Model evaluation completed successfully")

            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys)
