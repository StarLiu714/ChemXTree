# import sys
# sys.path.append("/ChemXTree/") 
from ChemXTree import ChemXTreePipeline

# Example usage
# Global Variables
DATASET_NAME = "BACE"
BASE_PATH = "/ChemXTree/Datasets/" + DATASET_NAME + "/"

# Set up the pipeline
GMFU_KWARGS = {
    "data_config": {
        "target": ["targets"],
        "categorical_cols": [],
        "num_workers": 4
    },
    "model_config": {
        "task": "classification",
        "learning_rate": 2.942395406229627e-07,
        "tree_depth": 4,
        "num_trees": 5,
        "chain_trees": True,
        "n_heads": 2,
        "residual_scale": 0.05,
        "gflu_stages": 3,
        "head_config": {
            "layers": "",
            "dropout": 0.3398222178487832,
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
        "batch_size": 16, 
        "gpus": 1,
        "early_stopping_min_delta": 0.001,
        "early_stopping_patience": 3,
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
            "weight_decay": 2.0919991735993723e-05
        },
        "lr_scheduler": "ReduceLROnPlateau",
        "lr_scheduler_params": {
            "mode": "min",
            "factor": 0.1,
            "patience": 10,
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
    weighted_loss_mu=17
    )
score = pipeline._run_gmfu_training()

print(f"ChemXTree GMF Unit AUCROC score: {score}")