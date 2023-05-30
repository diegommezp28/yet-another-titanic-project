import pandas as pd


class DataCleaning:
    """
    Class in charge of the cleaning operations we want to perform to
    our dataset
    """

    def __init__(self):
        self.fare_mean = 0.0
        self.fitted = False

    def fit(self, data: pd.DataFrame):
        """
        This will gather all the info needed for later transforming the data.
        This should be done with the training set.
        """
        self.fare_mean = data["Fare"].mean()
        self.fitted = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This will transformed the data based on the info previously fitted.
        """
        if not self.fitted:
            raise UnfittedException(
                "Transform was called on data cleaning before fitting the model."
            )

        transformed_data = data["Fare"].fillna(self.fare_mean)
        return transformed_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This will fit the DataCleaning model and then transform it using both times the received data.
        """
        self.fit(data)
        return self.transform(data)


class FeatureEnricher:
    """
    Class in charge of the feature engineering
    """

    def __init__(self):
        self.fitted = False

    def fit(self, data: pd.DataFrame):
        """
        This will gather all the info needed for later transforming the data.
        This should be done with the training set.
        """
        self.fitted = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This will transformed the data based on the info previously fitted.
        """
        if not self.fitted:
            raise UnfittedException(
                "Transform was called on data enricher before fitting the model."
            )

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This will fit the FeatureEnricher model and then transform it using both times the received data.
        """
        self.fit(data)
        return self.transform(data)


class UnfittedException(Exception):
    """Excpetion thrown when something tries to transform the data before fitting the classes."""
