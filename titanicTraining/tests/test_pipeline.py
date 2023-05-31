from titanicTraining.src import TrainModelPipeline
import pandas as pd
import numpy as np
import os 
import shutil


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

class TestTrainModelPipeline: # pragma: no cover

    def test_train(self, tmp_path):
        """
        This will test the main pipeline.
        """
        temp_data_folder = tmp_path / "titanic"
        os.mkdir(temp_data_folder)
        temp_train = str(temp_data_folder / "train.csv")
        temp_test = str(temp_data_folder / "test.csv")
        COMPLIANT_TRAIN.to_csv(temp_train)
        COMPLIANT_TEST.to_csv(temp_test)
        shutil.make_archive(temp_data_folder, format="zip", root_dir=temp_data_folder)
        zip_file_path = str(tmp_path / 'titanic.zip')

        TrainModelPipeline(zip_path=zip_file_path).run()
        TrainModelPipeline(train_path=temp_train, test_path=temp_test).run()