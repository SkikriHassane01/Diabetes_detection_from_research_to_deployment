import lightgbm as lgb
import numpy as np
from typing import Dict, Any, Optional
from src.models.base_model import BaseModel

class LightGBMModel(BaseModel):
    """
    LightGBM implementation for diabetes classification.
    """
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize LightGBM model with parameters.
        
        Args:
            model_params: Dictionary of model parameters
        """
        default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.01,
                'num_leaves': 20,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'objective': 'multiclass',
                'num_class': 3,
                'random_state': 42,
                'n_jobs': -1
        }
        
        # Update default parameters with provided parameters
        if model_params:
            default_params.update(model_params)
            
        super().__init__("LightGBM", default_params)
    
    def build(self) -> None:
        """Build the LightGBM model with specified parameters."""
        self.model = lgb.LGBMClassifier(**self.model_params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        if self.model is None:
            self.build()
        
        self.model.fit(
            X_train, 
            y_train,
            eval_metric='multi_logloss'
        )
        
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of feature importance ('split', 'gain')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        return dict(zip(
            range(self.model.n_features_), 
            self.model.feature_importances_
        ))