import os
import sys
import requests
from zipfile import ZipFile
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def download_data_from_github(self):
        try:
            logging.info("Downloading dataset from GitHub")

            os.makedirs(
                self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,
                exist_ok=True
            )

            response = requests.get(self.data_ingestion_config.GITHUB_ZIP_URL)

            if response.status_code != 200:
                raise Exception("Failed to download data from GitHub")

            with open(self.data_ingestion_config.ZIP_FILE_PATH, "wb") as f:
                f.write(response.content)

            logging.info("GitHub dataset downloaded successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def unzip_and_clean(self):
        try:
            logging.info("Unzipping the dataset")

            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, "r") as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)

            return (
                self.data_ingestion_config.DATA_ARTIFACTS_DIR,
                self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR
            )

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        try:
            logging.info("Starting data ingestion")

            self.download_data_from_github()
            imbalance_data_file_path, raw_data_file_path = self.unzip_and_clean()

            data_ingestion_artifacts = DataIngestionArtifacts(
                imbalance_data_file_path=imbalance_data_file_path,
                raw_data_file_path=raw_data_file_path
            )

            logging.info(f"Data ingestion completed: {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys)
