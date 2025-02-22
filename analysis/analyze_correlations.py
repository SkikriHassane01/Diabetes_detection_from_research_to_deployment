import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from typing import Tuple
from src.utils.logging_config import setup_logger
import warnings 
warnings.filterwarnings('ignore')

logger = setup_logger("Correlation analysis")

REPORT_DIST_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(__name__)), 'reports/correlations_figures'))

def analyze_correlations(data: pd.DataFrame, 
                        output_dir: str = REPORT_DIST_DIR,
                        target_column: str = 'Diabetes_012',
                        figsize: Tuple[int, int] = (12, 10)) -> pd.DataFrame:
    """
    Analyze and visualize feature correlations.
    
    Args:
        data: DataFrame containing the dataset
        output_dir: Directory to save generated figures
        target_column: Name of the target variable column
        figsize: Size of the figure for plotting
        
    Returns:
        DataFrame containing correlation matrix
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate correlation matrix
    correlation_matrix = data.corr()
    
    # TODO: Plot correlation heatmap
    plt.figure(figsize=figsize)
    
    # crate a mask to show only the lower triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
               mask=mask,  # Where mask=1: Hide value, Where mask=0: Show value  
               cmap='coolwarm', 
               annot=False,
               center=0, 
               square=True, 
               linewidths=.5, 
               cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300)
        logger.info(f'the Feature Correlation plot is saved in {output_dir}')
        plt.close()
    except Exception as e:
        logger.error(f"Failed to save the Feature Correlation matrix in {output_dir} ==> see the error {e}")
    
    # TODO: Plot correlations with target variable
    target_corrs = correlation_matrix[target_column].sort_values(ascending=False)
    target_corrs = target_corrs.drop(target_column)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=target_corrs.values, y=target_corrs.index, palette='viridis')
    plt.title(f'Feature Correlations with {target_column}')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.tight_layout()
    
    try:
        plt.savefig(os.path.join(output_dir, 'target_correlations.png'), dpi=300)
        logger.info(f'the target correlations plot is saved in {output_dir}')
        plt.close()
    except Exception as e:
        logger.error(f"Failed to save the target correlations in {output_dir} ==> see the error {e}")
        
    return correlation_matrix

# if __name__ == "__main__":
#     data = pd.read_csv("data/extracted/diabetes_data/diabetes_012_health_indicators_BRFSS2015.csv")
#     analyze_correlations(data)