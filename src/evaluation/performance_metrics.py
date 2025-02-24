import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from typing import Dict, Any, List
from src.utils.logging_config import setup_logger

logger = setup_logger('metrics')

class PerformanceMetrics:
    """
    Calculate and store performance metrics for diabetes binary classification models.
    """
    def __init__(self):
        self.class_names = ['No Diabetes', 'Diabetes']
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities for positive class
            
        Returns:
            Dictionary containing all performance metrics
        """
        metrics = {
            # Basic metrics
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            
            # Confusion matrix and report
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, 
                                                        target_names=self.class_names,
                                                        output_dict=True)
        }
        
        # Add AUC and average precision if probabilities are provided
        if y_prob is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_true, y_prob),
                'average_precision': average_precision_score(y_true, y_prob),
                'precision_recall_curve': precision_recall_curve(y_true, y_prob)
            })
        
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
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            results[name] = self.calculate_metrics(y_test, y_pred, y_prob)
            
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
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1'],
                'AUC-ROC': metrics.get('roc_auc', None)
            })
            
        return pd.DataFrame(summary_data)

    def get_binary_metrics_df(self, results: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Create DataFrames comparing specific binary classification metrics.
        
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
                    value = model_metrics['classification_report'][class_name][metric]
                    row[class_name] = value
                data.append(row)
            metric_dfs[metric] = pd.DataFrame(data)
            
        return metric_dfs

    def get_threshold_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, 
                            thresholds: np.ndarray = None) -> pd.DataFrame:
        """
        Calculate metrics at different probability thresholds.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            thresholds: Array of thresholds to evaluate
            
        Returns:
            DataFrame with metrics at each threshold
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)
            
        metrics_data = []
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            metrics_data.append({
                'threshold': threshold,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred)
            })
            
        return pd.DataFrame(metrics_data)