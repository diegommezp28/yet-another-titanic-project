import os
from glob import glob
import re

# def initialize_logs(self):
#     """
#     This handles the initialization of checkpoints and logging information
#     """

#     # All previous logs
#     log_folders = glob(LOG_DIR_BASE.format("*"))

#     def key_func(string):
#         """
#         Function to order the version folder paths based on their version number
#         """
#         match = re.findall(r'\d+$', string)
#         if len(match) > 0:
#             return int(match[0])
#         else:
#             return -1

#     # Calculates what number the current version should be even if some folder has been deleted
#     log_folders.sort(key=key_func, reverse=True)
#     if len(log_folders) == 0:
#         version = 0
#     else:
#         last_version_path = log_folders[0]
#         last_version = key_func(last_version_path)
#         version = last_version + 1

#     # Creates the new log and checkpoint folder for the current version
#     self.log_dir = LOG_DIR_BASE.format(version)
#     print("Logging Directory:", self.log_dir)
#     os.makedirs(self.log_dir)

#     # Initialize log dict from scratch for new runs or from a checkpoint to continue from a previous run.
#     self.continue_from_ckpt = False
#     self.ckpt_path = self.ckpt_path_param
#     if (not self.ckpt_path
#             or (self.ckpt_path == 'last' and last_version == -1)):
#         self.log_dict = {
#             "level": 0,
#             "epoch": 0,
#             "batch": 0,
#             "batch_size": self.batch_size,
#             "date": datetime.datetime.now(),
#             "loss": None,
#             "opti_state_dict": {},
#             "model_state_dict": {},
#             "from_rbgs": {},
#             "to_rgbs": {}
#         }
#     else:
#         # Continue from the specified checkpoint path or from the last checkpoint path
#         if self.ckpt_path == 'last':
#             # Find the greatest level checkpoint file in the last version folder
#             ckpt_paths = glob(os.path.join(
#                 last_version_path, 'lvl_*.ckpt'))
#             ckpt_paths = [path.replace(".ckpt", "") for path in ckpt_paths]
#             ckpt_paths.sort(key=key_func, reverse=True)
#             # TODO si el legnth es igual a 0 echar warning e ir al fallback
#             self.ckpt_path = ckpt_paths[0] + '.ckpt'


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
