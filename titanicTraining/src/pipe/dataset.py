from __future__ import annotations
import pandas as pd
import os
import shutil
import pandera as pa


DEFAULT_TEMP_UNZIP_DIR = os.path.join("temp", "data", "titanic")


class TitanicDataset:
    """
    This class will handle the ingestion of the raw data and its
    validation.
    """

    def __init__(self, train_path: str, test_path: str):
        """
        Reads and validates the training and testing data from paths.

            Parameters:
                train_path `str`: Path to the train file.
                test_path `str`: Path to the test file.
        """
        if DatasetValidator.validate_paths(train_path, test_path):
            unvalidated_train_data: pd.DataFrame = pd.read_csv(train_path)
            self.train_data: pd.DataFrame = DatasetValidator.validate_data_schema(
                unvalidated_train_data, "train"
            )

            unvalidated_test_data: pd.DataFrame = pd.read_csv(test_path)
            self.test_data: pd.DataFrame = DatasetValidator.validate_data_schema(
                unvalidated_test_data, "test"
            )

    @classmethod
    def create_from_zip(cls, zip_path: str) -> TitanicDataset:
        """
        Takes a zip file path, unzip it in a temporary location to
        read the data and deletes the temporary directory.

            Parameters:
                zip_path `str`: Path to the zip file containing the train and test data
            Returns:
                dataset `TitanicDataset`: Validated Dataset.
        """
        if os.path.isfile(zip_path):
            if zip_path.endswith(".zip"):
                shutil.unpack_archive(zip_path, DEFAULT_TEMP_UNZIP_DIR)
                dataset: TitanicDataset = TitanicDataset(
                    train_path=os.path.join(DEFAULT_TEMP_UNZIP_DIR, "train.csv"),
                    test_path=os.path.join(DEFAULT_TEMP_UNZIP_DIR, "test.csv"),
                )
                shutil.rmtree(DEFAULT_TEMP_UNZIP_DIR)
                return dataset
            else:
                raise DatasetIngestionException(
                    f"The specified file {zip_path} is not a .zip file."
                )
        else:
            raise DatasetIngestionException(
                f"The specified path {zip_path} does not contain a file."
            )


class DatasetValidator:
    """
    Class with a series of utility functions to validate the existence
    and correctness of the dataset that is trying to be read.
    """

    test_schema: pa.DataFrameSchema = pa.DataFrameSchema(
        columns={
            "PassengerId": pa.Column(
                int,
            ),
            "Pclass": pa.Column(int, pa.Check.isin([1, 2, 3])),
            "Name": pa.Column(str),
            "Sex": pa.Column(str, pa.Check.isin(["male", "female"])),
            "Age": pa.Column(
                float, pa.Check.greater_than_or_equal_to(0), nullable=True
            ),
            "SibSp": pa.Column(int, pa.Check.greater_than_or_equal_to(0)),
            "Parch": pa.Column(int, pa.Check.greater_than_or_equal_to(0)),
            "Ticket": pa.Column(str),
            "Fare": pa.Column(
                float, pa.Check.greater_than_or_equal_to(0), nullable=True
            ),
            "Cabin": pa.Column(str, nullable=True),
            "Embarked": pa.Column(str, pa.Check.isin(["S", "C", "Q"]), nullable=True),
        },
        unique=["PassengerId"],
    )
    train_schema: pa.DataFrameSchema = test_schema.add_columns(
        {
            "Survived": pa.Column(int, pa.Check.isin([0, 1])),
        }
    )

    @classmethod
    def validate_paths(cls, train_path: str, test_path: str) -> bool:
        """
        Utility method to validate the existence of the specify paths for both
        the training and testing dataset files.

            Parameters:
                train_path `str`: Path to the csv file containing the training data
                test_path `str`: Path to the csv file containing the testing data

        """
        if os.path.isfile(train_path):
            if train_path.endswith(".csv"):
                pass
            else:
                raise DatasetIngestionException(
                    f"The training file at {train_path} is not a CSV."
                )
        else:
            raise DatasetIngestionException(
                f"The training file at {train_path} does not exist."
            )

        if os.path.isfile(test_path):
            if test_path.endswith(".csv"):
                pass
            else:
                raise DatasetIngestionException(
                    f"The Testing file at {test_path} is not a CSV."
                )
        else:
            raise DatasetIngestionException(
                f"The testing file at {test_path} does not exist."
            )

        return True

    @classmethod
    def validate_data_schema(cls, data: pd.DataFrame, split: str = "test") -> bool:
        """
        This will validate the data compliance with the expected schema. This validation is crucial
        for the for the rest of the ML pipeline to function properly.

            Parameters:
                data `pandas.DataFrame`: Data to validate.
                split `str`: Either 'train' or 'test'. Used to chose the appropiate validation schema.

            Returns:
                validated_Data `pandas.DataFrame`: Validated Data.
        """
        if split == "test":
            return cls.test_schema(data)
        elif split == "train":
            return cls.train_schema(data)
        else:
            raise InvalidOptionException(
                f'{split} is an invalid option for split. Valid options are "train" or "test".'
            )


class DatasetIngestionException(Exception):
    """Exception thrown when there is an error reading the specified data source"""

class InvalidOptionException(Exception):
    """Thrown when an unexpected arguments was passed to a function."""
