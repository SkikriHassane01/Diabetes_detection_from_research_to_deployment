from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator
import mlflow
from src.utils.logging_config import setup_logger

logger = setup_logger('models')

class BaseModel(ABC):
    """
    Abstract base class for all models in the diabetes classification project.
    Enforces consistent interface across different model implementations.
    """
    def __init__(self, model_name: str, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.
        
        Args:
            model_name: Name identifier for the model
            model_params: Dictionary of model hyperparameters
        """
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model: Optional[BaseEstimator] = None
        logger.info(f"Initializing {model_name} with parameters: {model_params}")
        
    @abstractmethod
    def build(self) -> None:
        """
        Build the model with specified parameters.
        Must be implemented by concrete model classes.
        """
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model on provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions for each class.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict_proba(X)
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        try:
            mlflow.sklearn.save_model(self.model, path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving model to {path}: {str(e)}")
            raise
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        try:
            self.model = mlflow.sklearn.load_model(path)
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {str(e)}")
            raise
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.model_params
    
    def set_params(self, **params: Any) -> None:
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
        """
        self.model_params.update(params)
        if self.model is not None:
            self.model.set_params(**params)