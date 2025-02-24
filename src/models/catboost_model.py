from catboost import CatBoostClassifier
import numpy as np
from typing import Dict, Any, Optional, List
from src.models.base_model import BaseModel

class CatBoostModel(BaseModel):
    """
    CatBoost implementation for diabetes classification.
    """
    def __init__(self, 
                 model_params: Optional[Dict[str, Any]] = None,
                 cat_features: Optional[List[int]] = None):
        """
        Initialize CatBoost model with parameters.
        
        Args:
            model_params: Dictionary of model parameters
            cat_features: List of indices for categorical features
        """
        default_params = {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.01,
                'l2_leaf_reg': 5.0,
                'rsm': 0.8,
                'min_data_in_leaf': 20,
                'loss_function': 'MultiClass',
                'classes_count': 3,
                'random_seed': 42
        }
        
        # Update default parameters with provided parameters
        if model_params:
            default_params.update(model_params)
            
        super().__init__("CatBoost", default_params)
        self.cat_features = cat_features
    
    def build(self) -> None:
        """Build the CatBoost model with specified parameters."""
        self.model = CatBoostClassifier(**self.model_params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the CatBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        if self.model is None:
            self.build()
        
        self.model.fit(
            X_train, 
            y_train,
            cat_features=self.cat_features,
            plot=False
        )
        
    def get_feature_importance(self, type: str = 'FeatureImportance') -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            type: Type of feature importance 
                 ('FeatureImportance', 'PredictionValuesChange', etc.)
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        importance = self.model.get_feature_importance()
        return dict(enumerate(importance))