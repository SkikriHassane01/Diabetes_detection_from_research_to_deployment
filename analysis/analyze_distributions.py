import numpy as np
import pandas as pd
import os 
from pathlib import Path
import matplotlib.pyplot as plt 
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from src.utils.logging_config import setup_logger
import warnings 
warnings.filterwarnings('ignore')

logger = setup_logger("analyze_distribution")

REPORT_DIST_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(__name__)), 'reports/distributions_figures'))

def analyze_distributions(data: pd.DataFrame,
                        output_dir: str = REPORT_DIST_DIR,
                        target_column: str = 'diabetes',
                        continuous_features: List[str] = None,
                        binary_features: List[str] = None,
                        categorical_features: List[str] = None,
                        class_names: Dict = None,
                        figsize: tuple[int, int] = (15,10)) -> None:
    """
    Analyze and visualize the distributions of all features.
    
    Args:
        data: DataFrame containing the dataset
        output_dir: Directory to save generated figures
        target_column: Name of the target variable column
        continuous_features: List of continuous feature names
        binary_features: List of binary feature names
        categorical_features: List of categorical feature names
        class_names: Dictionary mapping class indices to names
        figsize: Size of the figure for plotting
    """
    
    # Create the output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Default feature lists
    if continuous_features is None:
        continuous_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    
    if binary_features is None:
        binary_features = ['hypertension', 'heart_disease']
        
    if categorical_features is None:
        categorical_features = ['gender', 'smoking_history']
    
    if class_names is None:
        class_names = {0: 'No Diabetes', 1: 'Diabetes'}
    
    # Plot categorical features distribution
    if categorical_features:
        n_rows = (len(categorical_features) + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4*n_rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        for i, feature in enumerate(categorical_features):
            sns.countplot(
                data=data,
                ax=axes[i],
                x=feature,
                palette='Set2'
            )
            
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            
            # Add count labels
            for p in axes[i].patches:
                axes[i].annotate(f'{int(p.get_height())}', 
                               (p.get_x() + p.get_width()/2., p.get_height()),
                               ha='center', va='bottom')
        
        # Hide unused subplots
        for j in range(len(categorical_features), len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(output_dir, 'categorical_features_distribution.png'), dpi=300)
            logger.info(f"The categorical feature distribution analysis is completed and saved in {output_dir}")
        except Exception as e:
            logger.error(f"An error occurred while saving the categorical plot into {output_dir} ==> exception {e}")
        plt.close(fig)

    # Plot continuous features with hist and boxplot
    if continuous_features:
        n_rows = len(continuous_features)
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.reshape(-1, 2) if len(continuous_features) > 1 else np.array([axes])
        
        for i, feature in enumerate(continuous_features):
            # Histogram
            sns.histplot(
                data=data,
                x=feature,
                kde=True,
                ax=axes[i,0],
                color='skyblue'
            )
            axes[i, 0].set_title(f"Distribution of {feature}")
            
            # Boxplot
            sns.boxplot(
                data=data,
                x=feature,
                ax=axes[i, 1],
                color='skyblue'
            )
            axes[i, 1].set_title(f"Boxplot of {feature}")
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(output_dir, 'continuous_features_distribution.png'), dpi=300)
            logger.info(f"The continuous feature distribution analysis is completed and saved in {output_dir}")
        except Exception as e:
            logger.error(f"An error occurred while saving the continuous features plot into {output_dir} ==> exception {e}")
        plt.close(fig)
    
    # Plot binary features distribution
    if binary_features:
        fig, axes = plt.subplots(1, len(binary_features), figsize=(6*len(binary_features), 5))
        axes = np.array([axes]) if len(binary_features) == 1 else axes
        
        for i, feature in enumerate(binary_features):
            sns.countplot(
                data=data,
                x=feature,
                ax=axes[i],
                palette=['skyblue', 'salmon']
            )
            
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            
            # Add count labels
            for p in axes[i].patches:
                axes[i].annotate(f'{int(p.get_height())}', 
                               (p.get_x() + p.get_width()/2., p.get_height()),
                               ha='center', va='bottom')
            
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(output_dir, 'binary_features_distribution.png'), dpi=300)
            logger.info(f"The binary feature distribution analysis is completed and saved in {output_dir}")
        except Exception as e:
            logger.error(f"An error occurred while saving the binary plot into {output_dir} ==> exception {e}")
        plt.close(fig)
        
if __name__ == "__main__":
    data_path = 'data/extracted/diabetes_prediction_dataset/diabetes_prediction_dataset.csv'
    data = pd.read_csv(data_path)
    analyze_distributions(data)