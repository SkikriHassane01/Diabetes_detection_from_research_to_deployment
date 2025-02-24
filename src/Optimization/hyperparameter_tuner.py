import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Callable
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, accuracy_score, f1_score, roc_auc_score
from src.utils.logging_config import setup_logger
import mlflow
import joblib
import os
from pathlib import Path

logger = setup_logger('hyperparameter_tuner')

class CatBoostHyperparameterTuner:
    """
    Hyperparameter optimization for CatBoost model using Optuna.
    Optimizes for a balance of recall and accuracy with a focus on the minority class.
    """
    
    def __init__(self, 
                 experiment_name: str = "catboost_hyperparameter_tuning",
                 model_dir: str = "models/optimized",
                 n_trials: int = 100,
                 cv_folds: int = 5,
                 random_state: int = 42,
                 recall_weight: float = 0.7,
                 accuracy_weight: float = 0.3):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            experiment_name: Name for MLflow experiment
            model_dir: Directory to save optimized model
            n_trials: Number of Optuna trials
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            recall_weight: Weight for recall in the objective function (0-1)
            accuracy_weight: Weight for accuracy in the objective function (0-1)
        """
        self.experiment_name = experiment_name
        self.model_dir = model_dir
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.recall_weight = recall_weight
        self.accuracy_weight = accuracy_weight
        
        # Ensure weights sum to 1
        total_weight = self.recall_weight + self.accuracy_weight
        self.recall_weight /= total_weight
        self.accuracy_weight /= total_weight
        
        # Initialize MLflow
        mlflow.set_experiment(experiment_name)
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info(f"Initialized CatBoostHyperparameterTuner with {n_trials} trials")
        logger.info(f"Optimization weights: Recall={recall_weight:.2f}, Accuracy={accuracy_weight:.2f}")
    
    def optimize(self, X: pd.DataFrame, y: pd.Series, categorical_features: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run the hyperparameter optimization process.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            categorical_features: List of indices for categorical features (if any)
            
        Returns:
            Dictionary with best parameters and study results
        """
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        
        def objective(trial):
            # Define the search space
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_seed': self.random_state,
                'verbose': 0,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC'
            }
            
            # Boosting type
            boosting_type = trial.suggest_categorical('boosting_type', ['Ordered', 'Plain'])
            params['boosting_type'] = boosting_type
            
            # Cross-validation scores
            cv_scores = self._cross_validate(X, y, params, categorical_features)
            
            # Calculate weighted objective (higher is better)
            weighted_score = (
                self.recall_weight * np.mean(cv_scores['recall']) +
                self.accuracy_weight * np.mean(cv_scores['accuracy'])
            )
            
            # Log metrics to MLflow
            with mlflow.start_run(nested=True):
                for metric_name, values in cv_scores.items():
                    mlflow.log_metric(f'mean_{metric_name}', np.mean(values))
                    mlflow.log_metric(f'std_{metric_name}', np.std(values))
                
                mlflow.log_metric('weighted_score', weighted_score)
                mlflow.log_params(params)
            
            return weighted_score
        
        # Create and run the study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        best_params = study.best_params
        
        # Save study for later analysis
        joblib.dump(study, os.path.join(self.model_dir, 'optuna_study.pkl'))
        
        logger.info(f"Optimization complete. Best score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def _cross_validate(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any], 
                       categorical_features: Optional[List[int]] = None) -> Dict[str, List[float]]:
        """
        Perform cross-validation with the given parameters.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            params: Model parameters to evaluate
            categorical_features: List of indices for categorical features (if any)
            
        Returns:
            Dictionary with lists of scores across folds
        """
        from catboost import CatBoostClassifier
        
        scores = {
            'accuracy': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model = CatBoostClassifier(**params)
            model.fit(
                X_train, y_train,
                cat_features=categorical_features,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                use_best_model=True,
                verbose=False
            )
            
            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            scores['accuracy'].append(accuracy_score(y_val, y_pred))
            scores['recall'].append(recall_score(y_val, y_pred))
            scores['f1'].append(f1_score(y_val, y_pred))
            scores['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
            
        return scores
    
    def train_final_model(self, X: pd.DataFrame, y: pd.Series, best_params: Dict[str, Any], 
                         categorical_features: Optional[List[int]] = None) -> Any:
        """
        Train final model with best parameters.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            best_params: Best parameters from optimization
            categorical_features: List of indices for categorical features (if any)
            
        Returns:
            Trained model
        """
        from catboost import CatBoostClassifier
        
        logger.info("Training final model with best parameters")
        
        # Train model with best parameters
        model = CatBoostClassifier(**best_params)
        model.fit(
            X, y,
            cat_features=categorical_features,
            verbose=False
        )
        
        # Save model
        model_path = os.path.join(self.model_dir, 'catboost_optimized.pkl')
        joblib.dump(model, model_path)
        logger.info(f"Saved optimized model to {model_path}")
        
        # Log final model to MLflow
        with mlflow.start_run():
            mlflow.log_params(best_params)
            mlflow.sklearn.log_model(model, "optimized_catboost_model")
        
        return model
    
    def visualize_optimization_results(self, study: optuna.study.Study) -> None:
        """
        Visualize optimization results from Optuna study.
        
        Args:
            study: Completed Optuna study
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create figures directory
            figures_dir = os.path.join(self.model_dir, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            
            # Optimization history
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'optimization_history.png'), dpi=300)
            
            # Parameter importances
            plt.figure(figsize=(12, 8))
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'parameter_importances.png'), dpi=300)
            
            # Parallel coordinate plot for top parameters
            plt.figure(figsize=(15, 8))
            optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'parallel_coordinate.png'), dpi=300)
            
            logger.info(f"Saved optimization visualizations to {figures_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")