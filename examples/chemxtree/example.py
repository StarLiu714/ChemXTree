from ChemXTree.ChemXTree_Pipeline import ChemXTreePipeline

# Example usage
# Global Variables
DATASET_NAME = "CYP2C9"
BASE_PATH = "/ChemXTree/Datasets/" + DATASET_NAME + "/"

# Set up the pipeline
MPNN_KWARGS = {
    "run_env": "local"  # "local" or "colab"
}
GMFU_KWARGS = {
    "data_config": {
        "categorical_cols": [],
    },
    "model_config": {
        "task": "classification",
        "metrics": ['accuracy', "auroc"],
        "metrics_params": [
            {"task": "multiclass", "num_classes": 2},
            {"task": "multiclass", "num_classes": 2}
        ]
    },
    "trainer_config": {
        "auto_lr_find": True,
        "checkpoints": None
    },
    "optimizer_config": {
        "optimizer": "AdamW",
        "optimizer_params": {},
        "lr_scheduler": "ReduceLROnPlateau",
        "lr_scheduler_params": {},
        "lr_scheduler_monitor_metric": "valid_loss"
    }
}


# Example usage
pipeline = ChemXTreePipeline(
    dataset_name=DATASET_NAME,
    base_path=BASE_PATH,
    mpnn_kwargs=MPNN_KWARGS,
    gmfu_kwargs=GMFU_KWARGS
    )
score = pipeline._run_ensemble()
print(f"ChemXTree AUCROC score: {score}")