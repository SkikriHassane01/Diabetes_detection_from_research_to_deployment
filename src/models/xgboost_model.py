import numpy as np
from typing import Dict, Any, Optional
from src.models.base_model import BaseModel
from xgboost import XGBClassifier

class XGBoostModel(BaseModel):
    """
    XGBoost implementation for diabetes binary classification.
    """
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model with parameters.
        
        Args:
            model_params: Dictionary of model parameters
        """
        default_params = {
            'objective': 'binary:logistic',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'scale_pos_weight': 1,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'eval_metric': 'logloss'  # Include eval_metric in model params instead of fit
        }
        
        if model_params:
            default_params.update(model_params)
            
        super().__init__("XGBoost", default_params)
    
    def build(self) -> None:
        """Build the XGBoost model with specified parameters."""
        self.model = XGBClassifier(**self.model_params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the XGBoost model."""
        if self.model is None:
            self.build()
            
        self.model.fit(X_train, y_train)
        
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of feature importance ('weight', 'gain', 'cover')
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.get_booster().get_score(importance_type=importance_type)