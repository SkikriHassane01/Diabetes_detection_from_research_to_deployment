import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from src.utils.logging_config import setup_logger
import warnings 
from pathlib import Path
warnings.filterwarnings('ignore')

logger = setup_logger("Feature_importance")

REPORT_DIST_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(__name__)), 'reports/feature_importance_figures'))
def analyze_feature_importance(data: pd.DataFrame,
                              output_dir: str = REPORT_DIST_DIR,
                              target_column: str = 'Diabetes_012',
                              n_estimators: int = 100,
                              top_k: int = 15,
                              figsize: Tuple[int, int] = (12, 8)) -> Dict:
    """
    Perform feature importance analysis using Random Forest.
    
    Args:
        data: DataFrame containing the dataset
        output_dir: Directory to save generated figures
        target_column: Name of the target variable column
        n_estimators: Number of trees in Random Forest
        top_k: Number of top features to display
        figsize: Size of the figure for plotting
        
    Returns:
        Dictionary containing feature importance scores
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Random Forest feature importance
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    
    try:
        rf.fit(X, y)
        rf_importances = pd.Series(rf.feature_importances_, index=X.columns)
        rf_importances = rf_importances.sort_values(ascending=False)
        
        # Plot Random Forest importances
        plt.figure(figsize=figsize)
        rf_importances[:top_k].plot(kind='bar', color='skyblue')
        plt.title('Feature Importance (Random Forest)')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'rf_feature_importance.png'), dpi=300)
        plt.close()
        logger.info(f"Successfully saving the feature importance plot in {output_dir}")

    except Exception as e:
        logger.error(f"Error in Random Forest feature importance calculation: {str(e)}")

# if __name__ == "__main__":
#     # Read the CSV file into a DataFrame first
#     data = pd.read_csv("data/extracted/diabetes_data/diabetes_012_health_indicators_BRFSS2015.csv")
#     analyze_feature_importance(data)