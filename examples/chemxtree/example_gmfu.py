from ChemXTree import ChemXTreePipeline

# Example usage
# Global Variables
DATASET_NAME = "BBBP"
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
        "n_heads": 2,
        "head_config": {
            "layers": "",
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
        "early_stopping_min_delta": 1e-5,
        "early_stopping_patience": 5,
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
            "patience": 10,
            "cooldown": 10,
            "verbose": True
        },
        "lr_scheduler_monitor_metric": "valid_loss"
    }
}

BAYESIAN_CONFIG = {
    'trainer_config': {
        'batch_size': [8, 16, 32] },
        'optimizer_config': {
            'optimizer_params': {
                'weight_decay': (1e-5, 1e-1)
                }   },
    'model_config': {
        'learning_rate':(1e-7, 1e-1),
        'tree_depth': (3, 9),
        'num_trees': (3, 9),
        'gflu_stages': (2, 5),
        'head_config': {
            'dropout': (0.1, 0.6)
            }
        }
    }

# Example usage
pipeline = ChemXTreePipeline(
    dataset_name=DATASET_NAME,
    base_path=BASE_PATH,
    gmfu_kwargs=GMFU_KWARGS,
    bayesian_optimization = True,
    bayesian_config = BAYESIAN_CONFIG,
    n_trials = 30
    )
score = pipeline._run_gmfu_training()

print(f"ChemXTree GMF Unit AUCROC score: {score}")