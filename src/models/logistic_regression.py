from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import Dict, Any, Optional
from src.models.base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression implementation for diabetes binary classification.
    """
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize Logistic Regression model with parameters.
        
        Args:
            model_params: Dictionary of model parameters
        """
        default_params = {
            'penalty': 'l2',
            'C': 1.0,
            'class_weight': 'balanced',
            'random_state': 42,
            'max_iter': 1000,
            'solver': 'lbfgs',
            'n_jobs': -1
        }
        
        if model_params:
            default_params.update(model_params)
            
        super().__init__("LogisticRegression", default_params)

    def build(self) -> None:
        """Build the Logistic Regression model with specified parameters."""
        self.model = LogisticRegression(**self.model_params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the Logistic Regression model."""
        if self.model is None:
            self.build()
        self.model.fit(X_train, y_train)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on coefficients."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return dict(enumerate(abs(self.model.coef_[0])))