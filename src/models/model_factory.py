from typing import Dict, Any, Optional
from src.models.base_model import BaseModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.catboost_model import CatBoostModel
from src.utils.logging_config import setup_logger

logger = setup_logger('model_factory')

class ModelFactory:
    """
    Factory class for creating model instances.
    """
    
    # Registry of available models
    _models = {
        'logistic_regression': LogisticRegressionModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'catboost': CatBoostModel,
    }
    
    @classmethod
    def get_model(cls, 
                  model_name: str, 
                  model_params: Optional[Dict[str, Any]] = None) -> BaseModel:
        """
        Create and return a model instance.
        
        Args:
            model_name: Name of the model to create
            model_params: Optional parameters for the model
            
        Returns:
            Instance of the requested model
            
        Raises:
            ValueError: If model_name is not recognized
        """
        model_class = cls._models.get(model_name.lower())
        
        if model_class is None:
            available_models = list(cls._models.keys())
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models are: {available_models}"
            )
        
        logger.info(f"Creating model: {model_name}")
        return model_class(model_params)
    
    @classmethod
    def list_available_models(cls) -> list:
        """
        List all available models.
        
        Returns:
            List of available model names
        """
        return list(cls._models.keys())