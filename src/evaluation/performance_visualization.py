import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os
from src.utils.logging_config import setup_logger

logger = setup_logger('visualization')
SAVE_DIR = 'reports/metrics_visualizations'

class PerformanceVisualizer:
    """
    Create visualizations for comparing training and testing performance metrics.
    """
    def __init__(self, save_dir: Optional[str] = SAVE_DIR):
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        self.colors = ['#2ecc71', '#e74c3c']  # Green for training, Red for testing


    def plot_final_comparison(self,
                          train_metrics: Dict,
                          test_metrics: Dict,
                          model_name: str = "Model",
                          figsize: tuple = (10, 6)) -> None:
        """
        Plot final training vs testing metrics comparison.
        
        Args:
            train_metrics: Training metrics from classification report
            test_metrics: Testing metrics from classification report
            model_name: Name of the model
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Extract metrics from classification reports
        metrics_to_plot = [
            ('Accuracy', 'accuracy'),
            ('Precision', 'macro avg', 'precision'),
            ('Recall', 'macro avg', 'recall'),
            ('F1 Score', 'macro avg', 'f1-score')
        ]
        
        # Prepare data
        labels = [m[0] for m in metrics_to_plot]
        train_values = []
        test_values = []
        
        for _, *keys in metrics_to_plot:
            # Navigate nested dictionaries
            train_value = train_metrics
            test_value = test_metrics
            for key in keys:
                train_value = train_value[key]
                test_value = test_value[key]
            train_values.append(train_value)
            test_values.append(test_value)

        x = np.arange(len(labels))
        width = 0.35

        # Plot bars
        plt.bar(x - width/2, train_values, width, label='Training', color=self.colors[0])
        plt.bar(x + width/2, test_values, width, label='Testing', color=self.colors[1])

        # Customize plot
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(f'{model_name} - Performance Comparison')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(train_values):
            plt.text(i - width/2, v, f'{v:.3f}', ha='center', va='bottom')
        for i, v in enumerate(test_values):
            plt.text(i + width/2, v, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        self._save_plot(f'{model_name.lower()}_final_comparison.png')

    def plot_confusion_matrix(self,
                            cm: np.ndarray,
                            is_training: bool = True,
                            figsize: tuple = (8, 6)) -> None:
        """
        Plot confusion matrix for either training or testing set.
        
        Args:
            cm: Confusion matrix array
            is_training: Whether this is training or testing data
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        data_type = "Training" if is_training else "Testing"
        class_names = ['No Diabetes', 'Prediabetes', 'Diabetes']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title(f'{data_type} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.tight_layout()
        self._save_plot(f'{data_type.lower()}_confusion_matrix.png')

    def plot_classification_reports(self,
                                train_report: Dict,
                                test_report: Dict,
                                model_name: str = "Model",
                                figsize: tuple = (15, 6)) -> None:
        """
        Plot classification reports for both training and testing sets side by side.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Function to prepare report data
        def prepare_report_data(report):
            df_report = pd.DataFrame(report).T
            # Remove 'support' column and 'accuracy' row for visualization
            df_plot = df_report.drop('support', axis=1)
            if 'accuracy' in df_plot.index:
                df_plot = df_plot.drop('accuracy', axis=0)
            return df_plot
        
        # Prepare data
        train_plot = prepare_report_data(train_report)
        test_plot = prepare_report_data(test_report)
        
        # Plot training report
        sns.heatmap(train_plot.astype(float),
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlGn',
                   vmin=0.0,
                   vmax=1.0,
                   ax=ax1,
                   cbar=False)
        ax1.set_title(f'{model_name} - Training Report')
        
        # Plot testing report
        sns.heatmap(test_plot.astype(float),
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlGn',
                   vmin=0.0,
                   vmax=1.0,
                   ax=ax2,
                   cbar=True)
        ax2.set_title(f'{model_name} - Testing Report')
        
        plt.tight_layout()
        self._save_plot(f'{model_name.lower()}_classification_reports.png')

    def plot_model_comparison(self,
                            summary_df: pd.DataFrame,
                            figsize: tuple = (14, 8)) -> None:
        plt.figure(figsize=figsize)
        
        # Match column names from _create_summary_df
        metrics = [
            'Train Accuracy', 
            'Test Accuracy', 
            'Train F1 (Macro)', 
            'Test F1 (Macro)'
        ]
        
        melted_df = summary_df.melt(
            id_vars='Model', 
            value_vars=metrics,
            var_name='Metric', 
            value_name='Score'
        )
        
        ax = sns.barplot(x='Metric', y='Score', hue='Model', 
                        data=melted_df, palette='viridis')
        
        plt.title('Model Performance Comparison')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value annotations
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points')
        
        plt.tight_layout()
        self._save_plot('model_comparison.png')
    
    def plot_per_class_comparison(self,
                                per_class_df: pd.DataFrame,
                                metric: str,
                                figsize: tuple = (12, 6)) -> None:
        """
        Plot per-class metric comparison across models.
        
        Args:
            per_class_df: DataFrame containing per-class metrics
            metric: Metric name being visualized (precision/recall/f1)
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        melted_df = per_class_df.melt(id_vars='Model', 
                                    var_name='Class', 
                                    value_name='Score')
        
        ax = sns.barplot(x='Class', y='Score', hue='Model', 
                    data=melted_df, palette='tab10')
        
        plt.title(f'Per-class {metric.capitalize()} Comparison')
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value annotations
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points')
        
        plt.tight_layout()
        self._save_plot(f'per_class_{metric}_comparison.png')
        
        
    def _save_plot(self, filename: str) -> None:
        """Save plot if save_dir is specified."""
        if self.save_dir:
            path = os.path.join(self.save_dir, filename)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {path}")
            plt.close()
