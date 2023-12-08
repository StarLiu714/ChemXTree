from ChemXTree import ChemXTreePipeline

# Example usage
# Global Variables
DATASET_NAME = "BBBP"
BASE_PATH = "/ChemXTree/Datasets/" + DATASET_NAME + "/"

# Set up the pipeline
MPNN_KWARGS = {
    "num_iters": 20,
    "ensemble_size": 5,
    "run_env": "local"  # "local" or "colab"
}

# Example usage
pipeline = ChemXTreePipeline(
    dataset_name=DATASET_NAME,
    base_path=BASE_PATH,
    mpnn_kwargs=MPNN_KWARGS
    )
score = pipeline._run_mpnn_pipeline()

print(f"MPNN Fingerprints are ready.")