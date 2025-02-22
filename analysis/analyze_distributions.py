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
def analyze_distributions(data : pd.DataFrame,
                          output_dir: str = REPORT_DIST_DIR,
                          target_column: str = 'Diabetes_012',
                          continuous_features: List[str] = None,
                          ordinal_features: List[str] = None,
                          class_names: Dict = None,
                          figsize: tuple[int, int] = (15,10)) -> None:
    """
    Analyze and visualize the distributions of all features.
    
    Args:
        data: DataFrame containing the dataset
        output_dir: Directory to save generated figures
        target_column: Name of the target variable column
        continuous_features: List of continuous feature names
        ordinal_features: List of ordinal feature names
        class_names: Dictionary mapping class indices to names
        figsize: Size of the figure for plotting
    """
    
    # create the output directory if not
    os.makedirs(output_dir, exist_ok=True)
    
    # create the default values 
    if continuous_features is None:
        continuous_features = ['BMI', 'MentHlth', 'PhysHlth']
    
    if ordinal_features is None:
        ordinal_features = ['GenHlth', 'Age', 'Education', 'Income']
    
    if class_names is None:
        class_names = {0: 'No Diabetes', 1: 'Prediabetes', 2: 'Diabetes'}
    
    binary_features = [col for col in data.columns if col not in continuous_features + ordinal_features + [target_column] and len(data[col].unique()) == 2]

    # TODO: Count plot for binary features with a 3 columns
    if binary_features:
        # Calculate rows needed
        n_rows = (len(binary_features) + 2) // 3
    
        # Adjust figure size based on number of features
        fig, axes = plt.subplots(n_rows, 3, figsize=(17, 4*n_rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        # Add spacing between subplots
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        for i, feature in enumerate(binary_features):
            
            sns.countplot(
                data = data,
                ax= axes[i],
                x = feature,
                palette=['skyblue', 'salmon']
            )
            
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].set_yticks([])  # Remove y-axis ticks
            axes[i].set_ylabel('')  # Remove y-axis label
            axes[i].set_xlabel('')  # Remove y-axis label
            for container in axes[i].containers:
                axes[i].bar_label(
                    container,
                    fmt="%d",
                    label_type = 'edge',
                    fontsize = 11,
                    fontweight = 'bold'
                )
                
        # Hide unused subplots
        for j in range(len(binary_features), len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(output_dir, 'binary_features_distribution.png'), dpi=300)
            logger.info(f"The binaries feature distribution analysis is completed and saved in {output_dir}")
        except Exception as e:
            logger.error(f"An error occur while saving the binary plot into the {output_dir} ==> exception {e}")
        plt.close(fig)
    
    # TODO: plot continuous features with hist and boxplot
    if continuous_features:
        n_rows = len(continuous_features)
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.reshape(-1, 2) if len(continuous_features) > 1 else np.array([axes])
        
        for i, feature in enumerate(continuous_features):
            # Histogram:
            sns.histplot(
                data = data,
                x = feature,
                kde= True,
                ax = axes[i,0],
                color='skyblue'
            )
            axes[i, 0].set_title(f"Distribution of {feature}")
            
            # Boxplot
            sns.boxplot(
                data = data,
                x = feature,
                ax = axes[i, 1],
                color='skyblue'
            )
            axes[i, 1].set_title(f"Boxplot of {feature}")
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(output_dir, 'continuous_features_distribution.png'), dpi=300)
            logger.info(f"The continuous feature distribution analysis is completed and saved in {output_dir}")
        except Exception as e:
            logger.error(f"An error occur while saving the continuous features plot into the {output_dir} ==> exception {e}")
        plt.close(fig)
    
    # TODO: plot count plot for ordinal features
    if ordinal_features:
        # Calculate rows needed
        n_rows = (len(ordinal_features) + 1) // 2
    
        # Adjust figure size based on number of features
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4*n_rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        # Add spacing between subplots
        plt.subplots_adjust(hspace=0.9, wspace=0.3)
        
        for i, feature in enumerate(ordinal_features):
            
            sns.countplot(
                data = data,
                ax= axes[i],
                x = feature,
                palette=['skyblue', 'salmon']
            )
            
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].set_yticks([])  # Remove y-axis ticks
            axes[i].set_ylabel('')  # Remove y-axis label
            axes[i].set_xlabel('')  # Remove x-axis label
            for container in axes[i].containers:
                axes[i].bar_label(
                    container,
                    fmt="%d",
                    label_type = 'edge',
                    fontsize = 10,
                )
                
        # Hide unused subplots
        for j in range(len(ordinal_features), len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(output_dir, 'ordinal_features_distribution.png'), dpi=300)
            logger.info(f"The ordinal feature distribution analysis is completed and saved in {output_dir}")
        except Exception as e:
            logger.error(f"An error occur while saving the ordinal plot into the {output_dir} ==> exception {e}")
        plt.close(fig)
        
# if __name__ == "__main__":
#     # Read the CSV file into a DataFrame first
#     data = pd.read_csv("data/extracted/diabetes_data/diabetes_012_health_indicators_BRFSS2015.csv")
#     analyze_distributions(data)