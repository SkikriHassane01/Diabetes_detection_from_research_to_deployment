from xgboost import XGBClassifier  # Changed import statement
import numpy as np
from typing import Dict, Any, Optional
from src.models.base_model import BaseModel

class XGBoostModel(BaseModel):
    """
    XGBoost implementation for diabetes classification.
    """
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model with parameters.
        
        Args:
            model_params: Dictionary of model parameters
        """
        default_params = {
               'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.01,
                'min_child_weight': 5,
                'gamma': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'scale_pos_weight': 1,
                'num_class': 3,
                'objective': 'multi:softprob',
                'random_state': 42,
                'n_jobs': -1
        }
        
        # Update default parameters with provided parameters
        if model_params:
            default_params.update(model_params)
            
        super().__init__("XGBoost", default_params)
    
    def build(self) -> None:
        """Build the XGBoost model with specified parameters."""
        self.model = XGBClassifier(**self.model_params)  # Changed from xgb.XGBClassifier
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        if self.model is None:
            self.build()
        
        self.model.fit(
            X_train, 
            y_train,
            verbose=False
        )
        
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature indices to importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Get feature importance scores
        importance_scores = self.model.feature_importances_
        return dict(enumerate(importance_scores))