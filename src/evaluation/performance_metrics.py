import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Any, List
from src.utils.logging_config import setup_logger

logger = setup_logger('metrics')

class PerformanceMetrics:
    """
    Calculate and store performance metrics for diabetes classification models.
    """
    def __init__(self):
        self.class_names = ['No Diabetes', 'Prediabetes', 'Diabetes']
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing all performance metrics
        """
        present_classes = np.unique(np.concatenate([y_true, y_pred]))
        self.class_names = [f'Class {cls}' for cls in present_classes]
        metrics = {
            # Basic metrics
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_precision': precision_score(y_true, y_pred, average='macro'),
            'macro_recall': recall_score(y_true, y_pred, average='macro'),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            
            # Per-class metrics "to see how well the model is performing for each class"
            'per_class_precision': precision_score(y_true, y_pred, average=None),
            'per_class_recall': recall_score(y_true, y_pred, average=None),
            'per_class_f1': f1_score(y_true, y_pred, average=None),
            
            # Confusion matrix and report
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, 
                                                        target_names=self.class_names,
                                                        output_dict=True)
        }
        
        return metrics

    def compare_models(self, 
                      models: Dict[str, object], 
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Compare multiple models' performance.
        
        Args:
            models: Dictionary of model name to trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing metrics for all models
        """
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating model: {name}")
            y_pred = model.predict(X_test)
            results[name] = self.calculate_metrics(y_test, y_pred)
            
        return results
    
    def get_summary_df(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create a summary DataFrame of basic metrics for all models.
        
        Args:
            results: Dictionary of model results from compare_models
            
        Returns:
            DataFrame with summary metrics
        """
        summary_data = []
        
        for model_name, metrics in results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['macro_precision'],
                'Recall': metrics['macro_recall'],
                'F1 Score': metrics['macro_f1']
            })
            
        return pd.DataFrame(summary_data)

    def get_per_class_df(self, results: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Create DataFrames of per-class metrics for all models.
        
        Args:
            results: Dictionary of model results from compare_models
            
        Returns:
            Dictionary containing DataFrames for each metric type
        """
        metric_dfs = {}
        metrics = ['precision', 'recall', 'f1']
        
        for metric in metrics:
            data = []
            for model_name, model_metrics in results.items():
                row = {'Model': model_name}
                for i, class_name in enumerate(self.class_names):
                    row[class_name] = model_metrics[f'per_class_{metric}'][i]
                data.append(row)
            metric_dfs[metric] = pd.DataFrame(data)
            
        return metric_dfs