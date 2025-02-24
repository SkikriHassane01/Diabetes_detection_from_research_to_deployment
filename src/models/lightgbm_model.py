import lightgbm as lgb
import numpy as np
from typing import Dict, Any, Optional
from src.models.base_model import BaseModel

class LightGBMModel(BaseModel):
    """
    LightGBM implementation for diabetes binary classification.
    """
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize LightGBM model with parameters.
        
        Args:
            model_params: Dictionary of model parameters
        """
        default_params = {
            'objective': 'xentropy', # binary # give more importance to the positive class
            'metric': 'recall', #binary_logloss
            'is_unbalance' : True,
            'scale_pos_weight': 2, # increase the weight of the positive class (diabetes)
            'boosting_type': 'gbdt',
            'learning_rate': 0.5,
            'num_leaves': 50,
            'max_depth': 10,
            'min_child_samples': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        if model_params:
            default_params.update(model_params)
            
        super().__init__("LightGBM", default_params)
    
    def build(self) -> None:
        """Build the LightGBM model with specified parameters."""
        self.model = lgb.LGBMClassifier(**self.model_params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the LightGBM model."""
        if self.model is None:
            self.build()
        
        self.model.fit(
            X_train, 
            y_train,
            eval_metric='binary_logloss'
        )
        
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of feature importance ('split', 'gain')
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        return dict(zip(
            range(self.model.n_features_), 
            self.model.feature_importances_
        ))