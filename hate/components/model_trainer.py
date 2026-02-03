import os
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import (
    ModelTrainerArtifacts,
    DataTransformationArtifacts,
)
from hate.ml.model import ModelArchitecture


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifacts: DataTransformationArtifacts,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def spliting_data(self, csv_path):
        try:
            logging.info("Reading transformed dataset")

            df = pd.read_csv(csv_path)

            x = (
                df[TWEET]
                .fillna("")
                .astype(str)
                .str.strip()
            )
            y = df[LABEL]

            mask = x != ""
            x = x[mask]
            y = y[mask]

            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.3,
                random_state=42,
            )

            logging.info(
                f"Split done | x_train={x_train.shape}, x_test={x_test.shape}"
            )

            return x_train, x_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)

    def tokenizing(self, x_train):
        try:
            logging.info("Starting tokenization")

            x_train = (
                x_train
                .fillna("")
                .astype(str)
                .str.strip()
            )
            x_train = x_train[x_train != ""]

            tokenizer = Tokenizer(
                num_words=self.model_trainer_config.MAX_WORDS,
                oov_token="<OOV>",
            )

            tokenizer.fit_on_texts(x_train)

            sequences = tokenizer.texts_to_sequences(x_train)

            sequences_matrix = pad_sequences(
                sequences,
                maxlen=self.model_trainer_config.MAX_LEN,
            )

            logging.info(
                f"Tokenization complete | sequence_shape={sequences_matrix.shape}"
            )

            return sequences_matrix, tokenizer

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        logging.info("Starting model trainer")

        try:
            x_train, x_test, y_train, y_test = self.spliting_data(
                csv_path=self.data_transformation_artifacts.transformed_data_path
            )

            model_architecture = ModelArchitecture()
            model = model_architecture.get_model()

            logging.info("Model architecture loaded")

            sequences_matrix, tokenizer = self.tokenizing(x_train)

            logging.info("Training started")

            model.fit(
                sequences_matrix,
                y_train,
                batch_size=self.model_trainer_config.BATCH_SIZE,
                epochs=self.model_trainer_config.EPOCH,
                validation_split=self.model_trainer_config.VALIDATION_SPLIT,
            )

            logging.info("Training completed")

            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)

            tokenizer_path = os.path.join(
                self.model_trainer_config.TRAINED_MODEL_DIR,
                "tokenizer.pickle",
            )

            with open(tokenizer_path, "wb") as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)

            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH, index=False)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH, index=False)

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path=self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path=self.model_trainer_config.Y_TEST_DATA_PATH,
            )

            logging.info("ModelTrainerArtifacts created successfully")

            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys)
