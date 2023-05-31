from abc import ABC, abstractmethod
from .dataset import TitanicDataset
from .preprocessing import DataCleaning, FeatureEnricher
from .train import Trainer
from rich import print
import warnings


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
        trainer_args_path: str = "",
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

    def ingest(self, continue_next=False):
        print("[yellow]Step: [/yellow]Ingesting")
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.trainer = Trainer(**self.trainer_kwargs)
            self.trainer.fit(
                self.dataset.train_data.iloc[:, 1:], self.dataset.train_data.iloc[:, 0]
            )
            self.next_step = self.evaluate
            if continue_next:
                self.next_step()

    def evaluate(self):
        print("[yellow]Step: [/yellow]Evaluating")
        evaluation_obj, evaluation_str = self.trainer.evaluate(
            self.dataset.train_data.iloc[:, 1:], self.dataset.train_data.iloc[:, 0]
        )
        print("\n\n[green]Train Set Metrics[/green]")
        print(evaluation_str)

    def run(self, continue_next=False):
        self.ingest(continue_next=True)

    def resume(self, step=None):
        self.next_step = step if step else self.next_step
        self.next_step(continue_next=True)
