# EXAMPLE
# Author: Star Liu <StarLiu@seas.upenn.edu>
"""Run for MPNN Pipeline"""
import sys
sys.path.append("/ChemXTree/")    
from ChemXTree.MPNN.mpnn.MPNN_Pipeline import MPNNPipeline


# Global Variables
DATASET_NAME = "CYP2C9"
BASE_PATH = "/ChemXTree/Datasets/" + DATASET_NAME + "/"

# For local
pipeline_local = MPNNPipeline(
    dataset_name=DATASET_NAME,
    base_path=BASE_PATH
)
pipeline_local.run_pipeline()

# For Colab environment
# Note: The following code only runs in the Google Colab environment
pipeline_colab = MPNNPipeline(
    dataset_name=DATASET_NAME,
    base_path=BASE_PATH,
    run_env='colab'
)
# Only uncomment the following and execute it in the Colab environment
# pipeline_colab.run_pipeline()
