# ChemXTree Pipeline
# Author: Star Liu <StarLiu@seas.upenn.edu>
# For license information, see LICENSE.TXT
"""ChemXTree Pipeline"""
import pandas as pd
from .GMFU import TabularModel


class ChemXTreePipeline:
    def __init__(
            self, 
            dataset_name: str = "",
            base_path: str = "./",
            mpnn_kwargs: dict = None, 
            gmfu_kwargs: dict = None,
            train_file: str = None, val_file: str = None, test_file: str = None,
            target_columns: list = ['targets'],
            num_classes: int = 2,
            bayesian_optimization: bool = False,
            bayesian_config: dict = None,
            n_trials: int = 20,
            weighted_loss_mu: int = 1,  # enabled only if bayesian_optimization is False
            save_best_model: bool = False
            ):
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.config_path = base_path + "optimized_config.json"
        self.checkpoints_dir = base_path + dataset_name + "_checkpoints"
        self.train_file = train_file if train_file is not None else f"{base_path}train{dataset_name}.csv"
        self.test_file = test_file if test_file is not None else f"{base_path}test{dataset_name}.csv"
        self.val_file = val_file if val_file is not None else f"{base_path}valid{dataset_name}.csv"
        self.mpnn_kwargs = mpnn_kwargs if mpnn_kwargs is not None else {}
        self.gmfu_kwargs = gmfu_kwargs if gmfu_kwargs is not None else {}
        self.target_cols = target_columns
        self.num_classes = num_classes
        self.bayesian_optimization = bayesian_optimization
        self.bayesian_config = bayesian_config
        self.n_trials = n_trials if self.bayesian_optimization else 1
        self.mu = weighted_loss_mu
        self.save_model = save_best_model

    def create_model(self, config_overrides=None):
        from .GMFU.model import GateModulationFeatureUnitConfig
        from .GMFU.configuration import DataConfig, OptimizerConfig, TrainerConfig
        from .GMFU.model.blocks.heads import LinearHeadConfig
        """Create and return the tabular model."""
        if config_overrides is None:
            config_overrides = {}

        # Define configurations using provided overrides
        data_config = DataConfig(
            continuous_cols=self.train_df.columns[:-1].tolist(),
            **{**self.gmfu_kwargs['data_config'], **config_overrides.get('data_config', {})}
            )
        trainer_config = TrainerConfig(
            **{**self.gmfu_kwargs['trainer_config'], **config_overrides.get('trainer_config', {})}
            )
        optimizer_config = OptimizerConfig(
            **{**self.gmfu_kwargs['optimizer_config'], **config_overrides.get('optimizer_config', {})}
            )
        model_config = GateModulationFeatureUnitConfig(
            **{**self.gmfu_kwargs['model_config'], **config_overrides.get('model_config', {})}
            )
        model_config.head_config = LinearHeadConfig(
            **{**self.gmfu_kwargs['model_config'].get('head_config', {}), **config_overrides.get('head_config', {})}
            ).__dict__

        # Create and return the model
        return TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
    
    def _run_mpnn_pipeline(self):
        from . import MPNNPipeline
        pipeline = MPNNPipeline(
            base_path=self.base_path,
            dataset_name=self.dataset_name,
            **self.mpnn_kwargs
        )
        pipeline.run_pipeline()
        print(f"MPNN Fingerprints are ready.")
    
    def load_fingerprint(self):
        """Load train, validation, and test data. Prepared for GMFU."""
        try:
            self.train_df = pd.read_csv(self.base_path + "train_fingerprint.csv").iloc[:, 1:]
            self.valid_df = pd.read_csv(self.base_path + "valid_fingerprint.csv").iloc[:, 1:]
            self.test_df = pd.read_csv(self.base_path + "test_fingerprint.csv").iloc[:, 1:]
        except FileNotFoundError as e:
            print(f"Error loading file: {e}")
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV file: {e}")

    def _run_gmfu_training(self):
        from .GMFU.utils import get_class_weighted_cross_entropy
        # Load data
        self.load_fingerprint()

        if self.bayesian_optimization:
            import optuna
            # Optuna study for Bayesian optimization
            study = optuna.create_study(
                direction="maximize")
            study.optimize(
                self.bayes_obj, n_trials=self.n_trials)
            # Print results of the best trial
            print("Best trial for GMFU:")
            trial = study.best_trial
            print("  Value: {}".format(trial.value))
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
            # Create and train the model using the best trial parameters
            model = self.create_model(config_overrides=trial.params)
            weighted_loss = get_class_weighted_cross_entropy(
                self.train_df[self.target_cols].values.ravel(), 
                mu=trial.suggest_int("mu", 0.05, 30))  

        else:
            model = self.create_model()
            weighted_loss = get_class_weighted_cross_entropy(
                self.train_df[self.target_cols].values.ravel(), mu=self.mu)

        # Fit model by re-defined parameters
        model.fit(
            train=self.train_df, validation=self.valid_df, loss=weighted_loss)
        # Predict on test set
        te_pred_df = model.predict(self.test_df)
        y_test, y_pred_test = te_pred_df[self.target_cols], te_pred_df.iloc[:, -2]
        # Evaluate model
        score = self.evaluate(y_test, y_pred_test)

        if self.save_model:
            roc_auc_str = str(score).replace(".", "_")
            model.save_model(f"model_auc_{roc_auc_str}")

        return score
    
    def _run_ensemble(self):
        """Run the ensemble of MPNN and GMFU models."""
        self._run_mpnn_pipeline()
        score = self._run_gmfu_training()
        print(f"Ensemble completed with score: {score}")
        return score
        
    def evaluate(self, targets, predictions, metric="auc"):
        """
        Evaluate the model based on the predictions and targets.
        Returns the AUC score on the test set.

        Args:
            targets (np.array): The true targets.
            predictions (np.array): The predictions of the model.
            metric (str, optional): The metric to use for evaluation. Defaults to 'auc'.

        Returns:
            float: The score of the model.
        """
        if metric == "auc":
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(targets, predictions)
        elif metric == "accuracy":
            from sklearn.metrics import accuracy_score
            return accuracy_score(targets, predictions)
        else:
            raise ValueError("Metric not supported.")
    
    def calculate_loss(self, model, df):
        """Calculates the loss by comparing prediction and target for the given dataset.
        Currently only available for binary classification.

        Args:
            model (TabularModel): The already trained model
            df (pd.DataFrame): The dataset to use for prediction.

        Returns:
            float: The cross entropy loss.
        """
        from sklearn.metrics import log_loss
        # Predict on the provided dataset
        pred_df = model.predict(df)
        y_true, y_pred = df[self.target_cols], pred_df.iloc[:, -self.num_classes*len(self.target_cols)]
    
        return log_loss(y_true, y_pred)

    def bayes_obj(self, trial):
        """Objective function for Bayesian optimization. 
        This hyperparameter optimization is only available for binary classification.

        Args:
            trial (optuna.trial): The trial object for Bayesian optimization.

        Returns:
            float: The loss of the model.
        """
        from .GMFU.utils import get_class_weighted_cross_entropy
        # Put external config into default config
        def get_config_value(key, default):
            return self.bayesian_config.get(key, default) if self.bayesian_config and key in self.bayesian_config else default

        # Config overrides for optimization
        config_overrides = {
            'trainer_config': {
                'batch_size': trial.suggest_categorical(
                    "batch_size", get_config_value('trainer_config', {'batch_size': [8, 16, 32]})['batch_size'])
                },
            'optimizer_config': {
                'optimizer_params': {
                    'weight_decay': 
                    trial.suggest_loguniform("weight_decay", *get_config_value('optimizer_config', {'optimizer_params': {'weight_decay': (1e-5, 1e-1)}})['optimizer_params']['weight_decay'])
                    }
                },
            'model_config': {
                'learning_rate': trial.suggest_loguniform(
                    "learning_rate", *get_config_value('model_config', {'learning_rate': (1e-7, 1e-1)})['learning_rate']),
                'tree_depth': trial.suggest_int(
                    "tree_depth", *get_config_value('model_config', {'tree_depth': (3, 9)})['tree_depth']),
                'num_trees': trial.suggest_int(
                    "num_trees", *get_config_value('model_config', {'num_trees': (3, 9)})['num_trees']),
                'gflu_stages': trial.suggest_int(
                    "gflu_stages", *get_config_value('model_config', {'gflu_stages': (2, 5)})['gflu_stages']),
                'head_config': {
                    'dropout': trial.suggest_uniform(
                        "dropout", *get_config_value('model_config', {'head_config': {'dropout': (0.1, 0.6)}})['head_config']['dropout'])
                    }
                }
            }
        # Create and train the model
        model = self.create_model(config_overrides)
        weighted_loss = get_class_weighted_cross_entropy(
            self.train_df[self.target_cols].values.ravel(), mu=trial.suggest_int("mu", 0.05, 30))
        model.fit(train=self.train_df, validation=self.valid_df, loss=weighted_loss)
        train_loss = self.calculate_loss(model, self.train_df)
        valid_loss = self.calculate_loss(model, self.valid_df)
        loss = 0.8 * valid_loss + 0.2 * train_loss  # assuming loss is of positive sign
        neg_loss = -loss
        # Apply penalty if loss > 0.25
        if loss > 0.25:
            penalty = -0.1
            neg_loss += penalty
        te_pred_df = model.predict(self.test_df)
        y_test, y_pred_test = te_pred_df[self.target_cols], te_pred_df.iloc[:, -2]
        te_score = self.evaluate(y_test, y_pred_test)
        print(f"Trial {trial.number} SCORE: {te_score}")

        return neg_loss

    def __call__(self, *args, **kwargs):
        """Run the pipeline."""
        result = self._run_ensemble()
        return result
