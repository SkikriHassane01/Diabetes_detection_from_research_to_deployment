import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Tuple
from pathlib import Path
from src.utils.logging_config import setup_logger
import warnings 
warnings.filterwarnings('ignore')

logger = setup_logger("Class imbalance analysis")

REPORT_DIST_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(__name__)), 'reports/class_imbalance_figures'))

def analyze_class_imbalance(data: pd.DataFrame, 
                           output_dir: str = REPORT_DIST_DIR,
                           target_column: str = 'diabetes',
                           class_names: Dict = None,
                           figsize: Tuple[int, int] = (10, 6)):
    """
    Analyze and visualize class imbalance in the target variable.
    
    Args:
        data: DataFrame containing the dataset
        output_dir: Directory to save generated figures
        target_column: Name of the target variable column
        class_names: Dictionary mapping class indices to names
        figsize: Size of the figure for plotting
        
    Returns:
        Dictionary containing class imbalance statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default class names if not provided
    if class_names is None:
        class_names = {0: 'No Diabetes', 1: 'Diabetes'}
    
    # Get target distribution
    class_counts = data[target_column].value_counts().sort_index()
    total_samples = len(data)
    class_proportions = class_counts / total_samples 
    
    # Calculate imbalance ratio
    imbalance_ratio = class_counts.max() / class_counts.min()
    
    # Create bar chart with counts and percentages
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=class_counts.index,
                     y=class_counts.values,
                     palette=['lightblue', 'salmon'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set class names as labels
    if hasattr(ax, 'set_xticklabels'):
        ax.set_xticklabels([class_names.get(i, i) for i in class_counts.index])
    
    # Add percentage and count annotations
    for i, (count, proportion) in enumerate(zip(class_counts, class_proportions)):
        plt.annotate(f'{count:,}\n({proportion:.1%})', 
                   xy=(i, count), 
                   xytext=(0, 5),  # Offset from point
                   textcoords='offset points',
                   ha='center', 
                   va='bottom',
                   fontsize=11)
    
    plt.title('Class Distribution in Target Variable')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    try:
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300)
        logger.info(f'The Class Distribution plot is saved in {output_dir}')
        plt.close()
    except Exception as e:
        logger.error(f"Failed to save Class Distribution plot in {output_dir} ==> see the error {e}")
    
        
if __name__ == "__main__":
    data_path = 'data/extracted/diabetes_prediction_dataset/diabetes_prediction_dataset.csv'
    data = pd.read_csv(data_path)
    analyze_class_imbalance(data)