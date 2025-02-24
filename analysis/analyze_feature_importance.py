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
                             target_column: str = 'diabetes',
                             n_estimators: int = 100,
                             top_k: int = 8,
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
    # Handle categorical variables
    df = data.copy()
    categorical_columns = ['gender', 'smoking_history']
    for col in categorical_columns:
        df[col] = pd.factorize(df[col])[0]
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Random Forest feature importance
    rf = RandomForestClassifier(n_estimators=n_estimators, 
                              random_state=42,
                              class_weight='balanced')
    
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
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for i, v in enumerate(rf_importances[:top_k]):
            plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'rf_feature_importance.png'), dpi=300)
        plt.close()
        logger.info(f"Successfully saving the feature importance plot in {output_dir}")

        return dict(rf_importances)

    except Exception as e:
        logger.error(f"Error in Random Forest feature importance calculation: {str(e)}")
        return None

if __name__ == "__main__":
    data_path = 'data/extracted/diabetes_prediction_dataset/diabetes_prediction_dataset.csv'
    data = pd.read_csv(data_path)
    analyze_feature_importance(data)