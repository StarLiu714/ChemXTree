# Smile to MPNN Fingerprint Pipeline
# Author: Star <Star@seas.hahaha.edu>
"""MPNN Pipeline, based on ChemProp"""
import subprocess
import pandas as pd
import os
from typing import Optional, List
import datetime


class MPNNPipeline:
    def __init__(
            self, 
            dataset_name: str = "",
            base_path: str = "./",
            run_env: str = "local",
            num_iters: int = 20,
            ensemble_size: int = 5,
            dataset_type: str = "classification",
            train_file: str = None, val_file: str = None, test_file: str = None,
            checkpoints_dir: str = None,
            target_columns: list = ['targets'],
            hidden_size: int = None,
            batch_size: int = None
            ):
        self.train_file = train_file if train_file is not None else f"{base_path}train{dataset_name}.csv"
        self.test_file = test_file if test_file is not None else f"{base_path}test{dataset_name}.csv"
        self.val_file = val_file if val_file is not None else f"{base_path}valid{dataset_name}.csv"
        self.base_path = base_path
        self.config_path = base_path + "optimized_config.json"
        self.checkpoints_dir = checkpoints_dir if checkpoints_dir is not None else base_path + dataset_name + "_checkpoints"
        self.num_iters = num_iters
        self.ensemble_size = ensemble_size
        self.dataset_type = dataset_type
        self.run_env = run_env
        self.target_cols = target_columns
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"{self.base_path}_{timestamp}.log"
    
    def __str__(self):
        return f"MPNNPipeline for dataset {self.dataset_name} in {self.run_env} environment."

    def __repr__(self):
        return (f"MPNNPipeline(dataset_name={self.dataset_name!r}, base_path={self.base_path!r}, "
                f"run_env={self.run_env!r}, num_iters={self.num_iters}, "
                f"ensemble_size={self.ensemble_size}, dataset_type={self.dataset_type!r})")

    def run_command(self, command):
        """Run a command in the specified environment and log output when run in Colab."""
        try:
            if self.run_env == "colab":
                with open(self.log_filename, "a") as log_file:
                    get_ipython().system(f'{command} | tee -a {log_file.name}')

            elif self.run_env == "local":
                subprocess.run(command, shell=True, check=True)

        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")

    def mpnn_hyperopt(self):
        """Run hyperparameter optimization for MPNN model
        """
        command = f"chemprop_hyperopt --dataset_type {self.dataset_type} --num_iters {self.num_iters} --data_path {self.train_file} --config_save_path {self.config_path}"
        self.run_command(command)

    def mpnn_train(self):
        """Train MPNN model
        """
        command = f"chemprop_train --data_path {self.train_file} --dataset_type {self.dataset_type} --ensemble_size {self.ensemble_size} --save_dir {self.checkpoints_dir} --config_path {self.config_path}"
        # If to specify hidden layer size and batch size, uncomment the following code
        command += f" --hidden_size {self.hidden_size}" if self.hidden_size is not None else ""
        command += f" --batch_size {self.batch_size}" if self.batch_size is not None else ""
        self.run_command(command)

    def mpnn_fingerprint(self, file_path, preds_path):
        """Generate MPNN fingerprints

        Args:
            file_path (str): path to the SMILES-targets file
            preds_path (str): path to save the generated fingerprints
        """
        command = f"chemprop_fingerprint --test_path {file_path} --checkpoint_dir {self.checkpoints_dir} --preds_path {preds_path}"
        self.run_command(command)

    def process_data(
            self, 
            file_path: str, drop_targets: bool = False, 
            save_path: Optional[str] = None, usecols: Optional[List[str]] = None) -> pd.DataFrame:
        """Process the data and save it to a csv file"""
        if not os.path.exists(file_path):
            raise ValueError(f"File path {file_path} does not exist.")
        df = pd.read_csv(file_path, usecols=usecols)
        if drop_targets:
            df.drop(columns=self.target_cols, inplace=True)
        if save_path:
            df.to_csv(f"{self.base_path}{save_path}", index=False)
        return df

    def merge_and_save(self, df1, df2, save_path):
        """Merge two dataframes and save it to a csv file"""
        merged_df = pd.concat([df1, df2], axis=1)
        merged_df.to_csv(f"{self.base_path}{save_path}", index=False)

    def run_pipeline(self):
        """Run the MPNN pipeline"""

        # Run hyperparameter optimization and training
        self.mpnn_hyperopt()
        self.mpnn_train()

        # Define fingerprint file paths
        train_fingerprint_path = f"{self.base_path}train_fingerprint.csv"
        test_fingerprint_path = f"{self.base_path}test_fingerprint.csv"
        val_fingerprint_path = f"{self.base_path}valid_fingerprint.csv"
        test_no_targets_path = f"{self.base_path}test_no_targets.csv"
        val_no_targets_path = f"{self.base_path}valid_no_targets.csv"

        # Process and merge fingerprint data for the training set
        self.mpnn_fingerprint(self.train_file, train_fingerprint_path)
        df_fp_train = self.process_data(train_fingerprint_path)
        df_targets_train = self.process_data(self.train_file, usecols=self.target_cols)
        self.merge_and_save(df_fp_train, df_targets_train, 'train_fingerprint.csv')

        # Process and save the test and validation data without targets
        self.process_data(self.test_file, drop_targets=True, save_path="test_no_targets.csv")
        self.process_data(self.val_file, drop_targets=True, save_path="valid_no_targets.csv")

        # Generate, process, and merge fingerprint data for the test set
        self.mpnn_fingerprint(test_no_targets_path, test_fingerprint_path)
        df_fp_test = self.process_data(test_fingerprint_path)
        df_targets_test = self.process_data(self.test_file, usecols=self.target_cols)
        self.merge_and_save(df_fp_test, df_targets_test, 'test_fingerprint.csv')

        # Generate, process, and merge fingerprint data for the validation set
        self.mpnn_fingerprint(val_no_targets_path, val_fingerprint_path)
        df_fp_val = self.process_data(val_fingerprint_path)
        df_targets_val = self.process_data(self.val_file, usecols=self.target_cols)
        self.merge_and_save(df_fp_val, df_targets_val, 'valid_fingerprint.csv')
                