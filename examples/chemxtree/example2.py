from ChemXTree.ChemXTree_Pipeline import ChemXTreePipeline

# Example usage
# Global Variables
DATASET_NAME = "CYP2C9"
BASE_PATH = "/ChemXTree/Datasets/" + DATASET_NAME + "/"

# Set up the pipeline
MPNN_KWARGS = {
    "num_iters": 1,
    "ensemble_size": 1,
    "run_env": "local"  # "local" or "colab"
}
GMFU_KWARGS = {
    "data_config": {
        "target": ["targets"],
        "categorical_cols": [],
        "num_workers": 4
    },
    "model_config": {
        "task": "classification",
        "n_heads": 2,
        "head_config": {
            "layers": "",
            "dropout": 0.1,
            "initialization": "kaiming"
        },
        "metrics": ['accuracy', "auroc"],
        "metrics_params": [
            {"task": "multiclass", "num_classes": 2},
            {"task": "multiclass", "num_classes": 2}
        ]
    },
    "trainer_config": {
        "auto_lr_find": True,
        "max_epochs": 10,
        "gpus": 1,
        "checkpoints": None,
        "trainer_kwargs": {
            "accelerator": "gpu",
            "devices": 1,
            "strategy": "dp",
            "num_nodes": 1
        }
    },
    "optimizer_config": {
        "optimizer": "AdamW",
        "optimizer_params": {},
        "lr_scheduler": "ReduceLROnPlateau",
        "lr_scheduler_params": {
            "mode": "min",
            "factor": 0.2,
            "patience": 5,
            "cooldown": 10,
            "verbose": True
        },
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
# Or alternatively, call the class
# score = pipeline()
print(f"ChemXTree AUCROC score: {score}")