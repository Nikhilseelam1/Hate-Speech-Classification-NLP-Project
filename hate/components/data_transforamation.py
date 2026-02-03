import os
import re
import sys
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download("stopwords")

from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifacts: DataIngestionArtifacts,
    ):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    def imbalance_data_cleaning(self):
        try:
            logging.info("Cleaning imbalance data")
            df = pd.read_csv(self.data_ingestion_artifacts.imbalance_data_file_path)

            df.drop(
                self.data_transformation_config.ID,
                axis=self.data_transformation_config.AXIS,
                inplace=True,
            )

            logging.info(f"Imbalance data cleaned | shape={df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def raw_data_cleaning(self):
        try:
            logging.info("Cleaning raw data")
            df = pd.read_csv(self.data_ingestion_artifacts.raw_data_file_path)

            df.drop(
                self.data_transformation_config.DROP_COLUMNS,
                axis=self.data_transformation_config.AXIS,
                inplace=True,
            )

            df.loc[df[self.data_transformation_config.CLASS] == 0,
                   self.data_transformation_config.CLASS] = 1

            df[self.data_transformation_config.CLASS].replace({2: 0}, inplace=True)

            df.rename(
                columns={
                    self.data_transformation_config.CLASS:
                    self.data_transformation_config.LABEL
                },
                inplace=True,
            )

            logging.info(f"Raw data cleaned | shape={df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def concat_dataframe(self):
        try:
            logging.info("Concatenating datasets")

            raw_df = self.raw_data_cleaning()
            imbalance_df = self.imbalance_data_cleaning()

            df = pd.concat([raw_df, imbalance_df], ignore_index=True)

            logging.info(f"Final dataset shape={df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def concat_data_cleaning(self, text):
        try:
            stemmer = SnowballStemmer("english")
            stop_words = set(stopwords.words("english"))

            text = str(text).lower()
            text = re.sub(r"\[.*?\]", "", text)
            text = re.sub(r"https?://\S+|www\.\S+", "", text)
            text = re.sub(r"<.*?>+", "", text)
            text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
            text = re.sub(r"\n", "", text)
            text = re.sub(r"\w*\d\w*", "", text)

            words = [
                stemmer.stem(word)
                for word in text.split()
                if word not in stop_words
            ]

            return " ".join(words)

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Starting data transformation")

            df = self.concat_dataframe()

            df = df.dropna(subset=[self.data_transformation_config.TWEET])

            df[self.data_transformation_config.TWEET] = (
                df[self.data_transformation_config.TWEET]
                .astype(str)
                .str.strip()
            )

            df = df[df[self.data_transformation_config.TWEET] != ""]

            df[self.data_transformation_config.TWEET] = df[
                self.data_transformation_config.TWEET
            ].apply(self.concat_data_cleaning)

            logging.info(f"Transformed data shape: {df.shape}")

            os.makedirs(
                self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                exist_ok=True,
            )

            df.to_csv(
                self.data_transformation_config.TRANSFORMED_FILE_PATH,
                index=False,
                header=True,
            )

            artifact = DataTransformationArtifacts(
                transformed_data_path=self.data_transformation_config.TRANSFORMED_FILE_PATH
            )

            logging.info("Data transformation completed successfully")
            return artifact

        except Exception as e:
            raise CustomException(e, sys)
