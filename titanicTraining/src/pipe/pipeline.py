from abc import ABC, abstractmethod
from .dataset import TitanicDataset
from .preprocessing import DataCleaning, FeatureEnricher
from .train import Trainer
from rich import print
import pandas as pd
from .utils import makedir, make_current_runs_folder
import warnings
import pickle
import os


class Pipeline(ABC):
    """
    Abstract class representing a Pipeline.
    """

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def resume(self, step):
        pass


class TrainModelPipeline(Pipeline):
    def __init__(
        self,
        train_path: str = "",
        test_path: str = "",
        zip_path: str = "",
        base_runs_folder: str = "runs",
        model_ckpt_name: str = "train_pipeline",
        **trainer_kwargs
    ):
        self.next_step = self.run
        self.dataset = None
        self.train_path = train_path
        self.test_path = test_path
        self.zip_path = zip_path
        self.data_cleaner = None
        self.feature_enricher = None
        self.trainer = None
        self.trainer_kwargs = trainer_kwargs
        self.model_ckpt_name = model_ckpt_name
        self.base_runs_folder = base_runs_folder
        self.initialize_folders()

    def initialize_folders(self):
        """
        Creates all the necessary folders for keeping the ckeckpoints
        """
        makedir(self.base_runs_folder)
        self.current_run_folder = make_current_runs_folder(self.base_runs_folder)

    def save(self, path):
        """
        Saves self in the fiven path
        """
        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()

    @classmethod
    def load(cls, path):
        """
        Loads instance object from the specified path
        """
        f = open(path, "rb")
        instance = pickle.load(f)
        f.close()
        return instance

    def create_and_save_ckpt(self, step_name: str):
        """
        Saves a checkpoint of self into the correct folder.
        """
        step_dir = os.path.join(self.current_run_folder, step_name)
        makedir(step_dir)
        ckpt_path = os.path.join(step_dir, self.model_ckpt_name + ".ckpt")
        self.save(ckpt_path)

    def ingest(self, continue_next=False):
        print("[yellow]Step: [/yellow]Ingesting")
        self.create_and_save_ckpt("ingest")
        self.dataset = (
            TitanicDataset(self.train_path, self.test_path)
            if not self.zip_path
            else TitanicDataset.create_from_zip(self.zip_path)
        )
        self.next_step = self.preprocessing
        if continue_next:
            self.next_step(continue_next=True)

    def preprocessing(self, continue_next=False):
        print("[yellow]Step: [/yellow]Preprocessing")
        self.create_and_save_ckpt("preprocessing")
        self.data_cleaner = DataCleaning()
        self.feature_enricher = FeatureEnricher()

        # Preprocess Training Data
        self.dataset.train_data = self.data_cleaner.fit_transform(
            self.dataset.train_data
        )
        self.dataset.train_data = self.feature_enricher.fit_transform(
            self.dataset.train_data
        )

        # Preprocess Testing Data
        self.dataset.test_data = self.data_cleaner.transform(self.dataset.test_data)
        self.dataset.test_data = self.feature_enricher.transform(self.dataset.test_data)

        self.next_step = self.train
        if continue_next:
            self.next_step(continue_next=True)

    def train(self, continue_next=False):
        print("[yellow]Step: [/yellow]Training")
        self.create_and_save_ckpt("train")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.trainer = Trainer(**self.trainer_kwargs)
            self.trainer.fit(
                self.dataset.train_data.iloc[:, 1:], self.dataset.train_data.iloc[:, 0]
            )
            self.next_step = self.evaluate
            if continue_next:
                self.next_step(
                    self.dataset.train_data.iloc[:, 1:],
                    self.dataset.train_data.iloc[:, 0],
                )

    def evaluate(self, X, y):
        print("[yellow]Step: [/yellow]Evaluating")
        self.create_and_save_ckpt("evaluate")
        evaluation_obj, evaluation_str = self.trainer.evaluate(X, y)
        print("\n\n[green]Evaluation Metrics[/green]")
        print(evaluation_str)

    def predict(self, X):
        """
        Performs a prediction from preprocessed data
        """
        prediction = self.trainer.predict(X)
        return prediction

    def tranform_predict(self, X: pd.DataFrame, inplace: bool = False):
        """
        Performs a prediction from unprocessed data.
        """
        X_aux = X if inplace else X.copy()
        X_aux = self.data_cleaner.transform(X_aux)
        X_aux = self.feature_enricher.transform(X_aux)
        return self.predict(X_aux)

    def run(self, continue_next=False):
        self.ingest(continue_next=True)

    def resume(self, step=None):
        """
        Resumes the training from the given step or from the next step in line.
        """
        self.initialize_folders()
        self.next_step = step if step else self.next_step
        self.next_step(continue_next=True)
