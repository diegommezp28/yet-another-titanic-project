from pipe import DataCleaning, FeatureEnricher
import pandas as pd
import numpy as np


TRAIN_COLS = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'] # pragma: no cover
TEST_COLS = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'] # pragma: no cover

COMPLIANT_TRAIN = pd.DataFrame.from_records([
    [1,0,3,"Braund, Mr. Owen Harris",'male',22.0,1,0,'A/5 21171',7.25,'C100','S'], 
    [2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",'female',38,1,0,'PC 17599',71.2833,'C85','C'],
    [3,1,3,"Heikkinen, Miss. Laina",'female',26,0,0,'STON/O2. 3101282',7.925,'C20','S']], 
    columns=TRAIN_COLS) # pragma: no cover

COMPLIANT_TEST = pd.DataFrame.from_records([
    [892,3,"Kelly, Mr. James",'male',34.5,0,0,'W.E.P. 330911',7.8292,np.NaN,'Q'],
    [893,3,"Wilkes, Mrs. James (Ellen Needs)",'female',47,1,0,'363272',7,np.NaN,'S'],
    [894,2,"Myles, Mr. Thomas Francis",'male',62,0,0,'240276',9.6875,np.NaN,'Q']
], columns=TEST_COLS) # pragma: no cover



class TestDataCleaning: # pragma: no cover
    def test_fit_transform_order(self):
        """
        This will check that we get the same results by doing fit_transform or
        fit and transform separately.
        """
        dc1 = DataCleaning()
        dc2 = DataCleaning()

        dc1.fit(COMPLIANT_TRAIN)
        train1 = dc1.transform(COMPLIANT_TRAIN)
        test1 = dc1.transform(COMPLIANT_TEST)

        train2 = dc2.fit_transform(COMPLIANT_TRAIN)
        test2 = dc2.transform(COMPLIANT_TEST)

        assert train1.equals(train2), "Train dataset is not the same when doing fit and transform separately and fit_transform"
        assert test1.equals(test2), "Test dataset is not the same when doing fit and transform separately and fit_transform"

        try:
            DataCleaning().transform(COMPLIANT_TRAIN)
            assert 1 == 0, "Should throw error when tranforming before fitting"
        except:
            pass

class TestDataEnricher: # pragma: no cover
    def test_fit_transform_order(self):
        """
        This will check that we get the same results by doing fit_transform or
        fit and transform separately.
        """
        dc = DataCleaning()
        train = dc.fit_transform(COMPLIANT_TRAIN)
        test = dc.transform(COMPLIANT_TEST)

        fe1 = FeatureEnricher()
        fe2 = FeatureEnricher()

        fe1.fit(train)
        train1 = fe1.transform(train)
        test1 = fe1.transform(test)

        train2 = fe2.fit_transform(train)
        test2 = fe2.transform(test)

        assert train1.equals(train2), "Train dataset is not the same when doing fit and transform separately and fit_transform"
        assert test1.equals(test2), "Test dataset is not the same when doing fit and transform separately and fit_transform"

        try:
            FeatureEnricher().transform(train)
            assert 1 == 0, "Should throw error when tranforming before fitting"
        except:
            pass

