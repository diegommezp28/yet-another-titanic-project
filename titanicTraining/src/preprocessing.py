import pandas as pd
import numpy as np
from typing import List


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

    def transform(self, data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        This will transformed the data based on the info previously fitted.
        """
        if not self.fitted:
            raise UnfittedException(
                "Transform was called on data cleaning before fitting the model."
            )

        transformed_data: pd.DataFrame = data if inplace else data.copy()
        transformed_data["Fare"] = data["Fare"].fillna(self.fare_mean)
        transformed_data["Embarked"] = transformed_data["Embarked"].fillna("S")
        del transformed_data["PassengerId"]
        return transformed_data

    def fit_transform(self, data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        This will fit the DataCleaning model and then transform it using both times the received data.
        """
        self.fit(data)
        return self.transform(data, inplace)


class FeatureEnricher:
    """
    Class in charge of the feature engineering
    """

    def __init__(self):
        self.fitted: bool = False
        self.age_nan_replace_proxy = None
        self.cabin_num_1 = None
        self.columns = [
            "Pclass",
            "Sex",
            "Embarked",
            "Ticket_Lett",
            "Cabin_Letter",
            "Name_Title",
            "Fam_Size",
        ]
        self.good_cols = {}

    def names(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This will generate features related with the 'Name' column in our data.
        """
        data["Name_Len"] = data["Name"].apply(lambda x: len(x))
        data["Name_Title"] = (
            data["Name"].apply(lambda x: x.split(",")[1]).apply(lambda x: x.split()[0])
        )
        del data["Name"]
        return data

    def age_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This will handle the nulls in the age column by replacing the nulls with the age mean
        of its 'Name_Title' + 'Pclass' group if it exists. The idea is that the status of a passenger
        is a good proxy for its age, so we try to find the passengers with most similar status
        by looking at other in the same class and with the same name title (Dr, Mr, Sr, Capt, etc).
        """
        # Best proxy for age is social status given by name_title + Pclass. We will try to replace as much as we can with just that
        data["Age_Null_Flag"] = data["Age"].apply(lambda x: 1 if pd.isnull(x) else 0)
        data["Age"] = self.age_nan_replace_proxy.transform(lambda x: x.fillna(x.mean()))
        return data

    def fam_size(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a categorical Fam_Size feature which takes into account the number of siblings, spouses,
        parents and children aboard the titanic.

        """
        data["Fam_Size"] = np.where(
            (data["SibSp"] + data["Parch"]) == 0,
            "Solo",
            np.where((data["SibSp"] + data["Parch"]) <= 3, "Nuclear", "Big"),
        )
        del data["SibSp"]
        del data["Parch"]
        return data

    def tickets(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates some features based on the ticket column.
        Adds ticket len as a feature and another categorical feature which encapsulates the
        information about the first letter of the ticket (which hopefully can have some info
        about the location of the cabin inside the titanic)
        """
        data["Ticket_Lett"] = data["Ticket"].apply(lambda x: str(x)[0]).astype(str)
        data["Ticket_Lett"] = np.where(
            (data["Ticket_Lett"]).isin(["1", "2", "3", "S", "P", "C", "A"]),
            data["Ticket_Lett"],
            np.where(
                (data["Ticket_Lett"]).isin(["W", "4", "7", "6", "L", "5", "8"]),
                "Low_ticket",
                "Other_ticket",
            ),
        )
        data["Ticket_Len"] = data["Ticket"].apply(lambda x: len(x))
        del data["Ticket"]
        return data

    def cabins(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts the first letter of the cabin code as a new feature.
        Adds categorical values wich indicates weather or not a cabin's number
        is in some quartil of the training cabin number's data
        """
        data["Cabin_Letter"] = data["Cabin"].apply(lambda x: str(x)[0])
        data["Cabin_num"] = pd.qcut(self.cabin_num_1, 3)

        # All this extra steps for the concat is to basically to not brake the inplace nature of this function
        temp = pd.concat(
            (data, pd.get_dummies(data["Cabin_num"], prefix="Cabin_num")),
            axis=1,
            copy=False,
            ignore_index=False,
        )
        data.drop(data.index[0:], inplace=True)
        data[temp.columns] = temp
        del data["Cabin"]
        del data["Cabin_num"]
        return data

    def dummies(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Converts unique values of certain columns into dummy columns (True, False)
        """
        for column in self.columns:
            data[column] = data[column].astype(str)
            data_good_cols = [
                i
                for i in self.good_cols[column]
                if i.replace(column + "_", "") in data[column].unique()
            ]
            data = pd.concat(
                (data, pd.get_dummies(data[column], prefix=column)[data_good_cols]),
                axis=1,
            )
            del data[column]
        return data

    def fit(self, data: pd.DataFrame, inplace: bool = False):
        """
        This will gather all the info needed for later transforming more data.
        This should be run once with the training set.
        """
        aux_data = data if inplace else data.copy()
        aux_data["Name_Title"] = (
            aux_data["Name"]
            .apply(lambda x: x.split(",")[1])
            .apply(lambda x: x.split()[0])
        )
        self.age_nan_replace_proxy = aux_data.groupby(["Name_Title", "Pclass"])["Age"]
        self.cabin_num_1 = data["Cabin"].apply(lambda x: str(x).split(" ")[-1][1:])
        self.cabin_num_1.replace("an", np.NaN, inplace=True)
        self.cabin_num_1 = self.cabin_num_1.apply(
            lambda x: int(x) if not pd.isnull(x) and x != "" else np.NaN
        )
        # Some preprocessing that need to be done to the training data before getting all the dummies
        aux_data = self.names(aux_data)
        aux_data = self.fam_size(aux_data)
        aux_data = self.tickets(aux_data)
        aux_data = self.cabins(aux_data)
        for column in self.columns:
            aux_data[column] = aux_data[column].astype(str)
            self.good_cols[column] = [
                column + "_" + i for i in aux_data[column].unique()
            ]

        self.fitted = True

    def transform(
        self, data: pd.DataFrame, inplace: bool = False, prev_fitted: bool = False
    ) -> pd.DataFrame:
        """
        This will transformed the data based on the info previously fitted.
        """
        if not self.fitted:
            raise UnfittedException(
                "Transform was called on data enricher before fitting the model."
            )

        transformed_data = data if inplace else data.copy()
        # If already fitted, avoid repreating some features already created
        if not prev_fitted:
            transformed_data = self.names(transformed_data)
            transformed_data = self.fam_size(transformed_data)
            transformed_data = self.tickets(transformed_data)
            transformed_data = self.cabins(transformed_data)

        transformed_data = self.age_input(transformed_data)
        transformed_data = self.dummies(transformed_data)
        return transformed_data

    def fit_transform(self, data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        This will fit the FeatureEnricher model and then transform it using both times the received data.
        """
        aux_data = data if inplace else data.copy()
        self.fit(aux_data, True)
        return self.transform(aux_data, True, True)


class UnfittedException(Exception):
    """Excpetion thrown when something tries to transform the data before fitting the classes."""
