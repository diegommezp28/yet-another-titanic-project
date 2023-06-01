import os
from glob import glob
import re

def make_current_runs_folder(base_folder: str) -> str:
    """
    Finds which run version should be the next and creates the correspondant
    folder which will hold all the checkpooints
    """
    run_folders = glob(os.path.join(base_folder, "run_*"))

    def key_func(string):
        """
        Function to order the version folder paths based on their version number
        """
        match = re.findall(r"\d+$", string)
        if len(match) > 0:
            return int(match[0])
        else:
            return -1

    run_folders.sort(key=key_func, reverse=True)
    if len(run_folders) == 0:
        version = 0
    else:
        last_version_path = run_folders[0]
        last_version = key_func(last_version_path)
        version = last_version + 1

    # Creates the checkpoint folder for the current version
    run_folder = os.path.join(base_folder, f"run_{version}")
    makedir(run_folder)
    return run_folder


def makedir(path: str):
    """
    Make dir if does not exist.
    """
    if not os.path.isdir(path):
        os.makedirs(path)
