from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import Dict, Any, Optional
from src.models.base_model import BaseModel

class RandomForestModel(BaseModel):
    """
    Random Forest implementation for diabetes binary classification.
    """
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize Random Forest model with parameters.
        
        Args:
            model_params: Dictionary of model parameters
        """
        default_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 4,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }
        
        if model_params:
            default_params.update(model_params)
            
        super().__init__("RandomForest", default_params)
    
    def build(self) -> None:
        """Build the Random Forest model with specified parameters."""
        self.model = RandomForestClassifier(**self.model_params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the Random Forest model."""
        if self.model is None:
            self.build()
        self.model.fit(X_train, y_train)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return dict(enumerate(self.model.feature_importances_))