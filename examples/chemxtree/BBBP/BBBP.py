import sys
sys.path.append("../../..") 
from ChemXTree import ChemXTreePipeline

# Example usage
# Global Variables
DATASET_NAME = "BBBP"
BASE_PATH = "../../../Datasets/" + DATASET_NAME + "/"

# Set up the pipeline
GMFU_KWARGS = {
    "data_config": {
        "target": ["targets"],
        "categorical_cols": [],
        "num_workers": 4
    },
    "model_config": {
        "task": "classification",
        "learning_rate": 1.1757136774823018e-06,
        "tree_depth": 6,
        "num_trees": 4,
        "chain_trees": True,
        "n_heads": 2,
        "residual_scale": 0.05,
        "gflu_stages": 3,
        "head_config": {
            "layers": "",
            "dropout": 0.22013378747973908,
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
        "max_epochs": 200,
        "batch_size": 32, 
        "gpus": 1,
        "early_stopping_min_delta": 0.001,
        "early_stopping_patience": 5,
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
        "optimizer_params": {
            "weight_decay": 0.0252342621486338
        },
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
    gmfu_kwargs=GMFU_KWARGS,
    weighted_loss_mu=6,
    save_best_model=True
    )
score = pipeline._run_gmfu_training()

print(f"ChemXTree GMF Unit AUCROC score: {score}")