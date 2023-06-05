import pandas as pd
import os
import shutil
from pipe import TitanicDataset, DatasetValidator, DatasetIngestionException, InvalidOptionException

TRAIN_COLS = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'] # pragma: no cover
TEST_COLS = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'] # pragma: no cover



COMPLIANT_TRAIN = pd.DataFrame.from_records([
    [1,0,3,"Braund, Mr. Owen Harris",'male',22.0,1,0,'A/5 21171',7.25,None,'S'], 
    [2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",'female',38,1,0,'PC 17599',71.2833,'C85','C'],
    [3,1,3,"Heikkinen, Miss. Laina",'female',26,0,0,'STON/O2. 3101282',7.925,None,'S']], 
    columns=TRAIN_COLS) # pragma: no cover

COMPLIANT_TEST = pd.DataFrame.from_records([
    [892,3,"Kelly, Mr. James",'male',34.5,0,0,'W.E.P. 330911',7.8292,None,'Q'],
    [893,3,"Wilkes, Mrs. James (Ellen Needs)",'female',47,1,0,'363272',7,None,'S'],
    [894,2,"Myles, Mr. Thomas Francis",'male',62,0,0,'240276',9.6875,None,'Q']
], columns=TEST_COLS) # pragma: no cover


UNCOMPLIANT_TRAIN = pd.DataFrame.from_records([
    [1,0,4,"Braund, Mr. Owen Harris",'males',22,1,0,'A/5 21171',7.25,None,'F'], 
    [2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",'female',38,1,0,'PC 17599',71.2833,'C85','C'],
    [3,1,-1,"Heikkinen, Miss. Laina",'femal',26,0,0,'STON/O2. 3101282',7.925,None,'S']], 
    columns=TRAIN_COLS) # pragma: no cover

UNCOMPLIANT_TEST = pd.DataFrame.from_records([
    [892,3,"Kelly, Mr. James",'male',34.5,0,-3,'330911',7.8292,None,'H'],
    [893,7,"Wilkes, Mrs. James (Ellen Needs)",'female',47,1,0,'363272',7,None,'S'],
    [894,-1,"Myles, Mr. Thomas Francis",'males',62,0,0,'240276',9.6875,None,'B']
], columns=TEST_COLS) # pragma: no cover


class TestTitanicDataset: # pragma: no cover
    def test_create_from_zip(self, tmp_path):
        """
        Check that the factory method create from zip acts the same as the usual constructor
        """
        temp_data_folder = tmp_path / "titanic"
        os.mkdir(temp_data_folder)
        temp_train = temp_data_folder / "train.csv"
        temp_test = temp_data_folder / "test.csv"
        COMPLIANT_TRAIN.to_csv(temp_train)
        COMPLIANT_TEST.to_csv(temp_test)
        shutil.make_archive(temp_data_folder, format="zip", root_dir=temp_data_folder)
        zip_file_path = tmp_path / 'titanic.zip'

        dataset_from_paths = TitanicDataset(train_path=str(temp_train), test_path=str(temp_test))
        dataset_from_zip = TitanicDataset.create_from_zip(str(zip_file_path))

        assert dataset_from_paths.train_data.equals(dataset_from_zip.train_data), 'Train data should be the same either created by zip or by files'
        assert dataset_from_paths.test_data.equals(dataset_from_zip.test_data), 'Test data should be the same either created by zip or by files'

        try:
            shutil.make_archive(temp_data_folder, format="tar", root_dir=temp_data_folder)
            TitanicDataset.create_from_zip(str(tmp_path / 'titanic.tar'))
            assert 1 == 0, "Create from zip should fail when passed another file extension other than zip"
        except DatasetIngestionException as e:
            assert 1 == 1

        try:
            TitanicDataset.create_from_zip(str(tmp_path / 'titanic.yaml'))
            assert 1 == 0,  "Create from zip should fail when passed an unexisting file"
        except DatasetIngestionException as e:
            assert 1 == 1

class TestDatasetValidator(): # pragma: no cover
    def test_validate_paths(self, tmp_path):
        """
        Checks if the dataset validator is validating the paths correctly.
        """
        temp_train = tmp_path / 'train.csv'
        temp_test = tmp_path / 'test.csv'
        COMPLIANT_TRAIN.to_csv(temp_train)
        COMPLIANT_TEST.to_csv(temp_test)
        COMPLIANT_TRAIN.to_csv(tmp_path / 'train.cv')

        try:
            DatasetValidator.validate_paths(train_path=str(tmp_path / 'train.ckpt'), test_path=str(temp_test))
            assert 1 == 0, "Validate paths should fail when train file does not exist"
        except DatasetIngestionException as e:
            pass

        try:
            DatasetValidator.validate_paths(train_path=str(tmp_path / 'train.cv'), test_path=str(temp_test))
            assert 1 == 0, "Validate paths should fail when train file is not csv"
        except DatasetIngestionException as e:
            pass

        try:
            DatasetValidator.validate_paths(train_path=str(temp_train), test_path=str(tmp_path / 'test.ckpt'))
            assert 1 == 0, "Validate paths should fail when test file does not exist"
        except DatasetIngestionException as e:
            pass

        try:
            DatasetValidator.validate_paths(train_path=str(temp_train), test_path=str(tmp_path / 'train.cv'))
            assert 1 == 0, "Validate paths should fail when test file is not csv"
        except DatasetIngestionException as e:
            pass

    

    def test_schema_validator(self, tmp_path):
        """
        Checks if the dataset validator is validating the schemas correctly.
        """
                
        try:
            DatasetValidator.validate_data_schema(UNCOMPLIANT_TEST, split='other')
            assert 1 == 0, "Schema validator should throw error on invalid split. Did not throw error on split='other'"
        except InvalidOptionException as e:
            pass

        try:
            DatasetValidator.validate_data_schema(COMPLIANT_TEST)
        except Exception as e:
            print(e)
            assert 1 == 0, "Error while validating schema compliant test set. Should not throw error"
        
        try:
            DatasetValidator.validate_data_schema(COMPLIANT_TRAIN, split='train')
        except Exception as e:
            print(e)
            assert 1 == 0, "Error while validating schema compliant train set. Should not throw error"  
        
        try:
            DatasetValidator.validate_data_schema(UNCOMPLIANT_TEST)
            assert 1 == 0, "Error while validating schema. This test schema should not be compliant"
        except Exception as e:
            pass
        
        try:
            DatasetValidator.validate_data_schema(UNCOMPLIANT_TRAIN, split='train')
            assert 1 == 0, "Error while validating schema. This train schema should not be compliant"
        except Exception as e:
            pass


