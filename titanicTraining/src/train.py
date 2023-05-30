from enum import Enum
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class TrainerModels:
    RandomForest = RandomForestClassifier

class Trainer:
    """
    Class in charge of training a model and keeping track of the arguments that were used
    for creating and training the model. 
    """
    def __init__(self, model: TrainerModels = TrainerModels.RandomForest, *model_args, **model_kwargs):
        self.model = model(*model_args, **model_kwargs)
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.trainer_args = None
        self.trainer_kwargs = None
    

    def fit(self, X, y, *trainer_args, **trainer_kwargs):
        """
        This will fit training data `X` and training labels `y` to the chosen model.
        The extra args and kwargs will be forwarded to the models fit method. 
        """
        self.trainer_args = trainer_args
        self.trainer_kwargs = trainer_kwargs
        self.model.fit(X, y, *trainer_args, **trainer_kwargs)
        return self.model
    
    def evaluate(self, X, y) -> dict:
        y_pred = self.predict(X)
        return classification_report(y, y_pred, output_dict=True)


    def predict(self, X):
        return self.model.predict(X)
    

    