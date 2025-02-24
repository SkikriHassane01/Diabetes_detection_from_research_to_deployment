import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import train_test_split, StratifiedKFold
import mlflow
import os
from datetime import datetime
from mlflow.models import infer_signature

from src.models.model_factory import ModelFactory
from src.evaluation.performance_metrics import PerformanceMetrics
from src.evaluation.performance_visualization import PerformanceVisualizer
from src.utils.logging_config import setup_logger

logger = setup_logger('training')

class DiabetesModelTrainer:
    """
    Comprehensive trainer for diabetes classification.
    Handles model training, evaluation, and performance visualization.
    """
    def __init__(self, 
                 experiment_name: str = "diabetes_classification",
                 visualization_dir: str = "reports/metrics_visualizations"):
        """
        Initialize the trainer.
        
        Args:
            experiment_name: Name for MLflow experiment
            visualization_dir: Directory for saving visualizations
        """
        self.metrics_calculator = PerformanceMetrics()
        
        # Ensure visualization directory exists
        os.makedirs(visualization_dir, exist_ok=True)
        self.visualizer = PerformanceVisualizer(save_dir=visualization_dir)
        
        self.model_factory = ModelFactory()
        
        # Initialize MLflow - ensure any active runs are closed
        self._ensure_no_active_run()
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"Initialized DiabetesModelTrainer with experiment: {experiment_name}")
        
    def _ensure_no_active_run(self):
        """Ensure no active MLflow run exists."""
        if mlflow.active_run():
            mlflow.end_run()
            
    def prepare_data(self, 
                    data: pd.DataFrame,
                    target_column: str = 'diabetes',
                    already_split: bool = False,
                    test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            already_split: Whether data already contains a 'split' column
            test_size: Proportion of test set
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Preparing data for training...")
        
        if already_split:
            if 'split' not in data.columns:
                logger.error("Data does not contain 'split' column despite already_split=True")
                raise ValueError("Data does not contain 'split' column despite already_split=True")
                
            train_df = data[data["split"] == "train"].copy()
            test_df = data[data["split"] == "test"].copy()
            
            if train_df.empty or test_df.empty:
                logger.error(f"Invalid split values. Train size: {len(train_df)}, Test size: {len(test_df)}")
                raise ValueError(f"Invalid split values. Train set or test set is empty.")
                
            # Drop split column if it exists
            if 'split' in train_df.columns:
                train_df = train_df.drop(columns=["split"])
            if 'split' in test_df.columns:
                test_df = test_df.drop(columns=["split"])

            # Split the data into features and target
            if target_column not in train_df.columns or target_column not in test_df.columns:
                logger.error(f"Target column '{target_column}' not found in data")
                raise ValueError(f"Target column '{target_column}' not found in data")
                
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
        else:
            # Validate target column exists
            if target_column not in data.columns:
                logger.error(f"Target column '{target_column}' not found in data")
                raise ValueError(f"Target column '{target_column}' not found in data")
                
            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                stratify=y, 
                random_state=42
            )
        
        logger.info(f"Data prepared successfully. Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
        
    def train_model(self,
                   model_name: str,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   model_params: Optional[Dict] = None,
                   cv_folds: int = 5) -> Dict:
        """
        Train a single model with cross-validation.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            model_params: Optional model parameters
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing trained model and results
        """
        logger.info(f"Starting training for {model_name}")
        
        # Ensure no active runs exist
        self._ensure_no_active_run()
        
        # Create unique run name
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_name = f"{model_name}-{timestamp}"
        
        try:
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                logger.info(f"MLflow run started: {run_id}")
                
                # Log dataset information
                mlflow.log_params({
                    "model_type": model_name,
                    "dataset_size": len(X_train),
                    "feature_count": X_train.shape[1],
                    "class_distribution": str(y_train.value_counts().to_dict()),
                    "cv_folds": cv_folds
                })
                
                # Create model
                model = self.model_factory.get_model(model_name, model_params)
                
                # Log model parameters
                if hasattr(model, 'get_params'):
                    model_params = model.get_params()
                    mlflow.log_params({f"param_{k}": v for k, v in model_params.items()})
                
                # Perform cross-validation
                cv_results = self._cross_validate(model, X_train, y_train, cv_folds)
                
                # Train final model on full training set
                model.train(X_train, y_train)
                
                # Get training predictions
                y_pred = model.predict(X_train)
                y_prob = self._get_probabilities(model, X_train)
                
                # Calculate training metrics
                train_metrics = self.metrics_calculator.calculate_metrics(y_train, y_pred, y_prob)
                
                # Log metrics to MLflow
                self._log_metrics_to_mlflow(train_metrics, 'train')
                self._log_cv_metrics_to_mlflow(cv_results)
                
                # Log feature names
                mlflow.log_param("feature_names", list(X_train.columns))
                
                # Log model artifact with signature
                try:
                    signature = infer_signature(X_train, y_pred)
                    mlflow.sklearn.log_model(
                        sk_model=model.model,
                        artifact_path="model",
                        signature=signature
                    )
                except Exception as e:
                    logger.warning(f"Could not log model artifact: {str(e)}")
                
                logger.info(f"Training completed for {model_name}. Accuracy: {train_metrics['accuracy']:.4f}")
                
                # Return results
                return {
                    'model': model,
                    'cv_results': cv_results,
                    'training_metrics': train_metrics,
                    'run_id': run_id
                }
                
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise

    def train_multiple_models(self,
                            models_config: Optional[List[Dict]] = None,
                            X_train: pd.DataFrame = None,
                            y_train: pd.Series = None,
                            cv_folds: int = 5) -> Dict[str, Dict]:
        """
        Train multiple models with specified configurations.
        
        Args:
            models_config: List of model configurations
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of model results
        """
        if X_train is None or y_train is None:
            logger.error("X_train and y_train must be provided")
            raise ValueError("X_train and y_train must be provided")
            
        # Validate input data
        if len(X_train) == 0 or len(y_train) == 0:
            logger.error("Empty training data provided")
            raise ValueError("Empty training data provided")
            
        # Default models if none specified
        models_config = models_config or [
            {'name': 'logistic_regression', 'params': None},
            {'name': 'random_forest', 'params': None},
            {'name': 'xgboost', 'params': None},
            {'name': 'lightgbm', 'params': None},
            {'name': 'catboost', 'params': None}
        ]
        
        logger.info(f"Training {len(models_config)} models on dataset of size {len(X_train)} with {X_train.shape[1]} features")
        
        results = {}
        successful_models = 0
        
        for config in models_config:
            model_name = config['name']
            model_params = config.get('params', None)
            
            # Skip any active runs before starting a new model
            self._ensure_no_active_run()
            
            try:
                logger.info(f"Starting training for {model_name}")
                model_result = self.train_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    model_params=model_params,
                    cv_folds=cv_folds
                )
                
                # Store result
                results[model_name] = model_result
                successful_models += 1
                logger.info(f"Successfully trained {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                # Store error information but don't include in results for evaluation
                results[model_name] = {
                    'model': None, 
                    'error': str(e),
                    'status': 'failed'
                }
        
        logger.info(f"Training complete: {successful_models}/{len(models_config)} models successful")
        return results

    def evaluate_models(self,
                       models_results: Dict[str, Dict],
                       X_test: pd.DataFrame,
                       y_test: pd.Series) -> Tuple[pd.DataFrame, Dict]:
        """
        Evaluate trained models and create visualizations.
        
        Args:
            models_results: Dictionary of trained models and their results
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Summary DataFrame and detailed metrics
        """
        logger.info("Evaluating models on test set...")
        evaluation_results = {}
        
        # Check for valid inputs
        if not isinstance(models_results, dict) or not models_results:
            logger.warning("No models to evaluate. Empty models_results dictionary.")
            return pd.DataFrame(), {}
            
        if X_test is None or y_test is None or len(X_test) == 0 or len(y_test) == 0:
            logger.error("Invalid test data provided")
            raise ValueError("Invalid test data provided")
        
        # Close any active runs
        self._ensure_no_active_run()
        
        # Collect valid models for evaluation
        valid_models = {
            name: results for name, results in models_results.items() 
            if 'model' in results and results['model'] is not None
        }
        
        if not valid_models:
            logger.warning("No valid models found for evaluation")
            return pd.DataFrame(), {}
            
        # Evaluate each model
        for model_name, results in valid_models.items():
            try:
                model = results['model']
                training_metrics = results.get('training_metrics', {})
                run_id = results.get('run_id')
                
                # Predict on test set
                y_pred = model.predict(X_test)
                y_prob = self._get_probabilities(model, X_test)
                
                # Calculate metrics
                test_metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred, y_prob)
                
                # Log test metrics to MLflow
                self._ensure_no_active_run()
                eval_run_name = f"eval_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                
                with mlflow.start_run(run_name=eval_run_name) as eval_run:
                    # Link to training run if available
                    if run_id:
                        mlflow.log_param("training_run_id", run_id)
                        
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("test_size", len(X_test))
                    self._log_metrics_to_mlflow(test_metrics, 'test')
                    
                    # Log confusion matrix as artifact if available
                    cm_path = os.path.join(self.visualizer.save_dir, f"{model_name}_confusion_matrix.png")
                    if os.path.exists(cm_path):
                        mlflow.log_artifact(cm_path)
                
                # Store evaluation results
                evaluation_results[model_name] = {
                    'train_metrics': training_metrics,
                    'test_metrics': test_metrics,
                    'cv_results': results.get('cv_results', {})
                }
                
                # Generate visualizations
                try:
                    self.visualizer.plot_model_performance_suite(
                        model_name=model_name,
                        train_metrics=training_metrics,
                        test_metrics=test_metrics
                    )
                    logger.info(f"Created visualizations for {model_name}")
                except Exception as viz_err:
                    logger.warning(f"Failed to create visualizations for {model_name}: {str(viz_err)}")
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {str(e)}")
        
        # Check if we have any successful evaluations
        if not evaluation_results:
            logger.warning("No models were successfully evaluated")
            return pd.DataFrame(), {}
        
        # Create comparative visualizations if we have multiple models
        if len(evaluation_results) > 1:
            try:
                self.visualizer.plot_models_comparison(evaluation_results)
                logger.info("Created model comparison visualizations")
            except Exception as e:
                logger.warning(f"Failed to create model comparison visualizations: {str(e)}")
        
        # Create summary DataFrame
        summary_df = self._create_summary_df(evaluation_results)
        
        # Log summary information
        self._ensure_no_active_run()
        with mlflow.start_run(run_name="evaluation_summary"):
            # Log best model if we can determine it
            if not summary_df.empty and 'Test F1' in summary_df.columns:
                try:
                    best_idx = summary_df['Test F1'].astype(float).idxmax()
                    best_model = summary_df.loc[best_idx, 'Model']
                    best_f1 = summary_df.loc[best_idx, 'Test F1']
                    
                    mlflow.log_param("best_model", best_model)
                    mlflow.log_param("best_f1_score", best_f1)
                    logger.info(f"Best model is {best_model} with F1 score of {best_f1:.4f}")
                except Exception as e:
                    logger.warning(f"Could not determine best model: {str(e)}")
            
            # Log summary table as artifact
            if not summary_df.empty:
                try:
                    summary_path = os.path.join(self.visualizer.save_dir, "model_summary.csv")
                    summary_df.to_csv(summary_path, index=False)
                    mlflow.log_artifact(summary_path)
                except Exception as e:
                    logger.warning(f"Failed to save summary CSV: {str(e)}")
        
        logger.info("Model evaluation completed")
        return summary_df, evaluation_results

    def _cross_validate(self,
                       model: Any,
                       X: pd.DataFrame,
                       y: pd.Series,
                       n_folds: int) -> Dict[str, List]:
        """
        Perform cross-validation.
        
        Args:
            model: Model to validate
            X: Feature data
            y: Target data
            n_folds: Number of CV folds
            
        Returns:
            Dictionary of CV metrics
        """
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        # Validate inputs
        if n_folds < 2:
            logger.warning(f"Invalid number of folds ({n_folds}), defaulting to 5")
            n_folds = 5
            
        if len(X) < n_folds:
            logger.warning(f"Too few samples ({len(X)}) for {n_folds} folds, reducing to 2 folds")
            n_folds = min(2, len(X))
        
        # Create stratified folds
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            try:
                # Get fold data
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model.train(X_fold_train, y_fold_train)
                
                # Predict and calculate metrics
                y_pred = model.predict(X_fold_val)
                y_prob = self._get_probabilities(model, X_fold_val)
                fold_metrics = self.metrics_calculator.calculate_metrics(y_fold_val, y_pred, y_prob)
                
                # Store scores
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    if metric in fold_metrics:
                        cv_scores[metric].append(fold_metrics[metric])
                
                if 'roc_auc' in fold_metrics:
                    cv_scores['auc'].append(fold_metrics['roc_auc'])
                
                logger.info(f"Fold {fold}/{n_folds} - Accuracy: {fold_metrics['accuracy']:.4f}, F1: {fold_metrics['f1']:.4f}")
                
            except Exception as e:
                logger.warning(f"Error in fold {fold}/{n_folds}: {str(e)}")
                # Continue with other folds
        
        # Log average metrics
        for metric, scores in cv_scores.items():
            if scores:  # Only if we have values
                logger.info(f"CV {metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                
        return cv_scores

    def _log_metrics_to_mlflow(self, metrics: Dict, prefix: str) -> None:
        """
        Log metrics to MLflow with prefix.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for metric names
        """
        if not metrics:
            logger.warning(f"No {prefix} metrics to log")
            return
            
        log_metrics = {}
        for key, value in metrics.items():
            # Skip non-numeric metrics like confusion matrix
            if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                log_metrics[f'{prefix}_{key}'] = float(value)
        
        mlflow.log_metrics(log_metrics)

    def _log_cv_metrics_to_mlflow(self, cv_results: Dict[str, List]) -> None:
        """
        Log cross-validation metrics to MLflow.
        
        Args:
            cv_results: Dictionary of CV metrics
        """
        if not cv_results:
            logger.warning("No CV results to log")
            return
            
        for metric, scores in cv_results.items():
            if scores:  # Only log if we have values
                mean_value = np.mean(scores)
                std_value = np.std(scores)
                
                # Check for valid values
                if not np.isnan(mean_value) and not np.isnan(std_value):
                    mlflow.log_metrics({
                        f'cv_mean_{metric}': float(mean_value),
                        f'cv_std_{metric}': float(std_value)
                    })

    def _get_probabilities(self, model: Any, X: Union[pd.DataFrame, np.ndarray]) -> Optional[np.ndarray]:
        """
        Safely get probability predictions from a model.
        
        Args:
            model: The trained model
            X: Input features
            
        Returns:
            Array of probabilities for positive class or None if not available
        """
        if not hasattr(model, 'predict_proba'):
            return None
            
        try:
            probs = model.predict_proba(X)
            # Handle both 1D and 2D probability arrays
            if probs.ndim == 2:
                return probs[:, 1]  # Return probabilities for positive class
            elif probs.ndim == 1:
                return probs  # Return as is if already 1D
            else:
                logger.warning(f"Unexpected probability array shape: {probs.shape}")
                return None
        except Exception as e:
            logger.warning(f"Error getting probabilities: {str(e)}")
            return None

    def _create_summary_df(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create summary DataFrame of results.
        
        Args:
            results: Dictionary of model results
            
        Returns:
            DataFrame with summary metrics
        """
        if not results:
            logger.warning("No results to summarize")
            return pd.DataFrame()
            
        summary_data = []
        
        for model_name, metrics in results.items():
            # Skip if missing essential metrics
            if 'train_metrics' not in metrics or 'test_metrics' not in metrics:
                logger.warning(f"Skipping {model_name} in summary - missing metrics")
                continue
                
            train_metrics = metrics['train_metrics']
            test_metrics = metrics['test_metrics']
            cv_results = metrics.get('cv_results', {})
            
            # Create row with essential model metrics
            row_data = {'Model': model_name}
            
            # Add metrics with validation
            for metric_name, source_dict, prefix in [
                ('accuracy', train_metrics, 'Train'), 
                ('accuracy', test_metrics, 'Test'),
                ('precision', train_metrics, 'Train'), 
                ('precision', test_metrics, 'Test'),
                ('recall', train_metrics, 'Train'), 
                ('recall', test_metrics, 'Test'),
                ('f1', train_metrics, 'Train'), 
                ('f1', test_metrics, 'Test')
            ]:
                if metric_name in source_dict:
                    row_data[f'{prefix} {metric_name.capitalize()}'] = source_dict[metric_name]
            
            # Add CV metrics if available
            if cv_results and 'accuracy' in cv_results and cv_results['accuracy']:
                row_data['CV Accuracy (mean ± std)'] = f"{np.mean(cv_results['accuracy']):.3f} ± {np.std(cv_results['accuracy']):.3f}"
            
            if cv_results and 'f1' in cv_results and cv_results['f1']:
                row_data['CV F1 (mean ± std)'] = f"{np.mean(cv_results['f1']):.3f} ± {np.std(cv_results['f1']):.3f}"
            
            # Add ROC-AUC if available
            if 'roc_auc' in test_metrics:
                row_data['ROC-AUC'] = test_metrics['roc_auc']
            
            summary_data.append(row_data)
        
        # Create and return DataFrame
        if summary_data:
            df = pd.DataFrame(summary_data)
            logger.info(f"Created summary dataframe with columns: {df.columns.tolist()}")
            return df
        else:
            logger.warning("No valid data for summary dataframe")
            return pd.DataFrame()