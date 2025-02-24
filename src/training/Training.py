import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
from datetime import datetime

from src.models.model_factory import ModelFactory
from src.evaluation.performance_metrics import PerformanceMetrics
from src.evaluation.performance_visualization import PerformanceVisualizer
from src.utils.logging_config import setup_logger

logger = setup_logger('training')

class DiabetesModelTrainer:
    """
    Trainer for diabetes classification models with MLflow tracking.
    """
    def __init__(self, experiment_name: str = "diabetes_classification"):
        """
        Initialize the trainer.
        
        Args:
            experiment_name: Name for MLflow experiment
        """
        self.metrics_calculator = PerformanceMetrics()
        self.visualizer = PerformanceVisualizer(save_dir='reports/metrics_visualizations')
        self.model_factory = ModelFactory()
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)

    def prepare_data(self, 
                    data: pd.DataFrame,
                    target_column: str = 'Diabetes_012',
                    already_split: bool = False,
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of test set
            
        Returns:
            Training and test sets
        """
        logger.info("Preparing data for training...")
        
        if already_split:
            train_df = data[data["split"] == "train"].drop(columns=["split"])
            test_df = data[data["split"] == "test"].drop(columns=["split"])

            # Split the data into features and target
            X_train = train_df.drop(columns=["Diabetes_012"])
            y_train = train_df["Diabetes_012"]

            X_test = test_df.drop(columns=["Diabetes_012"])
            y_test = test_df["Diabetes_012"]
            
            logger.info(f"Data split completed . Training set size: {len(X_train)}, Test set size: {len(X_test)}")
            return X_train, X_test, y_train, y_test
        else:
            # Separate features and target
            X = data.drop(target_column, axis=1)
            y = data[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                stratify=y, 
                random_state=42
            )
            
            logger.info(f"Data split completed. Training set size: {len(X_train)}, Test set size: {len(X_test)}")
            return X_train, X_test, y_train, y_test

    def train_models(self, 
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    models_to_train: Optional[List[str]] = None) -> Dict:
        """
        Train specified models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            models_to_train: List of model names to train
            
        Returns:
            Dictionary of trained models
        """
        if models_to_train is None:
            models_to_train = ['random_forest', 'lightgbm', 'catboost']
            
        trained_models = {}
        
        for model_name in models_to_train:
            logger.info(f"Training {model_name}...")
            
            with mlflow.start_run(nested=True) as run:
                # Create and train model
                model = self.model_factory.get_model(model_name)
                model.train(X_train, y_train)
                
                # Get training predictions
                train_preds = model.predict(X_train)
                
                # Calculate training metrics
                train_report = classification_report(y_train, train_preds, output_dict=True)
                
                # Log metrics to MLflow
                mlflow.log_metrics({
                    'train_accuracy': train_report['accuracy'],
                    'train_precision_macro': train_report['macro avg']['precision'],
                    'train_recall_macro': train_report['macro avg']['recall'],
                    'train_f1_macro': train_report['macro avg']['f1-score']
                })
                
                # Log model parameters if available
                if hasattr(model, 'get_params'):
                    mlflow.log_params(model.get_params())
                
                trained_models[model_name] = {
                    'model': model,
                    'training_metrics': {
                        'confusion_matrix': confusion_matrix(y_train, train_preds),
                        'classification_report': train_report
                    }
                }
                
                logger.info(f"{model_name} training completed. Accuracy: {train_report['accuracy']:.4f}")
        
        return trained_models

    def evaluate_models(self,
                       models: Dict,
                       X_test: pd.DataFrame,
                       y_test: pd.Series) -> Tuple[pd.DataFrame, Dict]:
        """
        Evaluate trained models and create visualizations.
        
        Args:
            models: Dictionary of trained models and their training metrics
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Summary DataFrame and detailed metrics
        """
        logger.info("Evaluating models on test set...")
    
        results = {}
        for model_name, model_info in models.items():
            model = model_info['model']
            training_metrics = model_info['training_metrics']
            
            # Get test predictions
            test_preds = model.predict(X_test)
            
            # Calculate comprehensive metrics using PerformanceMetrics
            full_test_metrics = self.metrics_calculator.calculate_metrics(y_test, test_preds)
            
            results[model_name] = {
                'train_metrics': training_metrics,
                'test_metrics': {
                    'confusion_matrix': full_test_metrics['confusion_matrix'],
                    'classification_report': full_test_metrics['classification_report'],
                    'full_metrics': full_test_metrics
                }
            }
            
            # Create all visualizations
            self.visualizer.plot_final_comparison(
                train_metrics=training_metrics['classification_report'],
                test_metrics=full_test_metrics['classification_report'],
                model_name=model_name
            )
            
            self.visualizer.plot_classification_reports(
                train_report=training_metrics['classification_report'],
                test_report=full_test_metrics['classification_report'],
                model_name=model_name
            )
            
            # Log comprehensive metrics to MLflow
            with mlflow.start_run(nested=True) as run:
                # Log macro metrics
                mlflow.log_metrics({
                    'test_accuracy': full_test_metrics['accuracy'],
                    'test_precision_macro': full_test_metrics['macro_precision'],
                    'test_recall_macro': full_test_metrics['macro_recall'],
                    'test_f1_macro': full_test_metrics['macro_f1']
                })
                
                # Log per-class metrics
                for i, class_name in enumerate(self.metrics_calculator.class_names):
                    mlflow.log_metrics({
                        f'test_precision_{class_name}': full_test_metrics['per_class_precision'][i],
                        f'test_recall_{class_name}': full_test_metrics['per_class_recall'][i],
                        f'test_f1_{class_name}': full_test_metrics['per_class_f1'][i]
                    })

        # Create comprehensive summary
        summary_df = self._create_summary_df(results)
        
        # Add new comparison visualization
        self.visualizer.plot_model_comparison(summary_df)
        
        # Create per-class comparison
        per_class_dfs = self.metrics_calculator.get_per_class_df(
            {name: res['test_metrics']['full_metrics'] for name, res in results.items()}
        )
        for metric, df in per_class_dfs.items():
            self.visualizer.plot_per_class_comparison(df, metric)
        
        logger.info("Model evaluation and visualization completed.")
        return summary_df, results

    def _create_summary_df(self, results: Dict) -> pd.DataFrame:
        summary_data = []
        
        for model_name, metrics in results.items():
            train_report = metrics['train_metrics']['classification_report']
            test_metrics = metrics['test_metrics']['full_metrics']
            
            summary_data.append({
                'Model': model_name,
                'Train Accuracy': train_report['accuracy'],
                'Test Accuracy': test_metrics['accuracy'],
                'Train F1 (Macro)': train_report['macro avg']['f1-score'],
                'Test F1 (Macro)': test_metrics['macro_f1']
            })
        
        return pd.DataFrame(summary_data)