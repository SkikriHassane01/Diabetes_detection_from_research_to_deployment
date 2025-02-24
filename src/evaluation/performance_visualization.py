import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import os
from src.utils.logging_config import setup_logger

logger = setup_logger('visualization')

class PerformanceVisualizer:
    """
    Comprehensive visualization suite for model performance analysis.
    Provides paired training/testing visualizations and detailed metric comparisons.
    """
    def __init__(self, save_dir: Optional[str] = 'reports/metrics_visualizations'):
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        self.colors = sns.color_palette("viridis", n_colors=8)
        self.metrics = ['precision', 'recall', 'f1-score']
        self.class_names = ['No Diabetes', 'Diabetes']
        
    def plot_model_performance_suite(self, 
                                   model_name: str,
                                   train_metrics: Dict[str, Any],
                                   test_metrics: Dict[str, Any],
                                   figsize: Tuple[int, int] = (20, 15)) -> None:
        """
        Generate comprehensive performance visualization suite for a single model.
        """
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f'Performance Analysis - {model_name}', y=1.02, size=16)

        # Create a 2x2 grid for the main plots
        gs = plt.GridSpec(2, 2, figure=fig)

        # Classification Reports
        self._plot_classification_reports(
            train_metrics['classification_report'],
            test_metrics['classification_report'],
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1])
        )

        # Confusion Matrix
        cm_ax = fig.add_subplot(gs[1, 0])
        self._plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            "Test Set Confusion Matrix",
            cm_ax
        )

        # ROC Curve or Metrics Bar Plot
        metrics_ax = fig.add_subplot(gs[1, 1])
        if 'roc_auc' in test_metrics:
            self._plot_comparison_metrics(
                train_metrics, test_metrics,
                ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                metrics_ax
            )
        else:
            self._plot_comparison_metrics(
                train_metrics, test_metrics,
                ['accuracy', 'precision', 'recall', 'f1'],
                metrics_ax
            )

        plt.tight_layout()
        
        if self.save_dir:
            path = os.path.join(self.save_dir, f'{model_name.lower()}_performance.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance suite to {path}")
        plt.close()

    def plot_models_comparison(self, evaluation_results: Dict[str, Dict]) -> None:
        """Generate comparative visualization of multiple models' performance."""
        fig = plt.figure(figsize=(15, 10))
        
        # Create metrics comparison plot
        ax = fig.add_subplot(111)
        metrics_data = []
        
        for model_name, results in evaluation_results.items():
            test_metrics = results['test_metrics']
            metrics_data.append({
                'Model': model_name,
                'Accuracy': test_metrics['accuracy'],
                'Precision': test_metrics['precision'],
                'Recall': test_metrics['recall'],
                'F1 Score': test_metrics['f1'],
                'ROC-AUC': test_metrics.get('roc_auc', None)
            })
        
        df = pd.DataFrame(metrics_data)
        df_melted = df.melt('Model', var_name='Metric', value_name='Score')
        
        sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', ax=ax)
        ax.set_title('Model Performance Comparison')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if self.save_dir:
            path = os.path.join(self.save_dir, 'models_comparison.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved models comparison to {path}")
        plt.close()

    def _plot_classification_reports(self,
                                   train_report: Dict,
                                   test_report: Dict,
                                   ax1: plt.Axes,
                                   ax2: plt.Axes) -> None:
        """Plot paired training and testing classification reports."""
        def prepare_report_data(report: Dict) -> pd.DataFrame:
            data = []
            for class_name in self.class_names + ['macro avg', 'weighted avg']:
                if class_name in report:
                    row = [report[class_name][metric] for metric in self.metrics]
                    data.append(row)
            return pd.DataFrame(data, 
                              columns=self.metrics,
                              index=self.class_names + ['macro avg', 'weighted avg'])
        
        train_df = prepare_report_data(train_report)
        test_df = prepare_report_data(test_report)
        
        sns.heatmap(train_df, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=0, vmax=1, ax=ax1, cbar=False)
        sns.heatmap(test_df, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=0, vmax=1, ax=ax2, cbar=True)
        
        ax1.set_title('Training Report')
        ax2.set_title('Testing Report')

    def _plot_confusion_matrix(self,
                             cm: np.ndarray,
                             title: str,
                             ax: plt.Axes) -> None:
        """Plot a single confusion matrix."""
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names,
                    ax=ax, cbar=True)
        
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    def _plot_comparison_metrics(self,
                               train_metrics: Dict,
                               test_metrics: Dict,
                               metrics: List[str],
                               ax: plt.Axes) -> None:
        """Plot comparison of train and test metrics."""
        data = []
        for metric in metrics:
            if metric in train_metrics and metric in test_metrics:
                data.append({
                    'Metric': metric,
                    'Train': train_metrics[metric],
                    'Test': test_metrics[metric]
                })
        
        df = pd.DataFrame(data)
        df_melted = df.melt('Metric', var_name='Dataset', value_name='Score')
        
        sns.barplot(data=df_melted, x='Metric', y='Score', hue='Dataset', ax=ax)
        ax.set_title('Performance Metrics Comparison')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)

    def _save_plot(self, filename: str) -> None:
        """Save plot if save_dir is specified."""
        if self.save_dir:
            path = os.path.join(self.save_dir, filename)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {path}")
            plt.close()