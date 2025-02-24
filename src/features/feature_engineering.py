import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.logging_config import setup_logger

logger = setup_logger('feature_engineering')

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Handles feature engineering for the diabetes classification project.
    Inherits from sklearn's BaseEstimator and TransformerMixin for pipeline compatibility.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'age_ranges': {
                1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39',
                5: '40-44', 6: '45-49', 7: '50-54', 8: '55-59',
                9: '60-64', 10: '65-69', 11: '70-74', 12: '75-79',
                13: '80+'
            },
            'health_features': ['GenHlth', 'MentHlth', 'PhysHlth'],
            'risk_factors': ['HighBP', 'HighChol', 'HeartDiseaseorAttack'],
            'lifestyle_features': ['PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'Smoker'],
        }
        logger.info("FeatureEngineer initialized with configuration")

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering transformation")
        df = X.copy()
        
        # Create basic composite features
        df = self._create_health_score(df)
        df = self._create_risk_score(df)
        df = self._create_lifestyle_score(df)
        
        # Create interaction features
        df = self._create_age_bmi_interaction(df)
        df = self._create_health_risk_interaction(df)
        
        df = self._create_healthcare_access_index(df)
        
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        return df

    def _create_health_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite health score from health-related features"""
        try:
            # Normalize each component to 0-1 scale
            df['HealthScore'] = (
                # GenHlth is 1-5 where 1 is excellent, so we invert it
                (5 - df['GenHlth']) / 4 +
                # MentHlth and PhysHlth are days of poor health (0-30)
                (30 - df['MentHlth']) / 30 +
                (30 - df['PhysHlth']) / 30
            ) / 3  # Average the components
            
            logger.info("Created HealthScore feature")
            return df
        except Exception as e:
            logger.error(f"Error creating health score: {str(e)}")
            raise

    def _create_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk score from risk factors"""
        try:
            # Weight different risk factors
            df['RiskScore'] = (
                df['HighBP'] * 2.0 +  # Higher weight for blood pressure
                df['HighChol'] * 1.5 +  # Medium weight for cholesterol
                df['HeartDiseaseorAttack'] * 1.5 +  # Medium weight for heart conditions
                (df['BMI'] > 30).astype(int) * 1.5 +  # Medium weight for obesity
                (df['Age'] >= 7).astype(int) * 1.5  # Medium weight for age > 50
            ) / 8.0  # Normalize to 0-1 scale
            
            logger.info("Created RiskScore feature")
            return df
        except Exception as e:
            logger.error(f"Error creating risk score: {str(e)}")
            raise

    def _create_lifestyle_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite lifestyle score from lifestyle-related features"""
        try:
            # Positive factors add to score, negative factors subtract
            df['LifestyleScore'] = (
                df['PhysActivity'] * 1.0 +  # Physical activity is positive
                df['Fruits'] * 0.5 +        # Healthy diet is positive
                df['Veggies'] * 0.5 -       
                df['HvyAlcoholConsump'] * 1.0 -  # Heavy alcohol consumption is negative
                df['Smoker'] * 1.0          # Smoking is negative
            ) / 3.0  # Normalize to roughly -1 to 1 scale
            
            logger.info("Created LifestyleScore feature")
            return df
        except Exception as e:
            logger.error(f"Error creating lifestyle score: {str(e)}")
            raise

    def _create_age_bmi_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction between age and BMI"""
        try:
            # Normalize BMI by its mean to keep the scale reasonable
            df['AgeBMI'] = df['Age'] * df['BMI'] / df['BMI'].mean()
            logger.info("Created Age-BMI interaction feature")
            return df
        except Exception as e:
            logger.error(f"Error creating age-BMI interaction: {str(e)}")
            raise

    def _create_health_risk_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction between health score and risk factors"""
        try:
            if 'HealthScore' in df.columns and 'RiskScore' in df.columns:
                df['HealthRiskInteraction'] = df['HealthScore'] * df['RiskScore']
                logger.info("Created Health-Risk interaction feature")
            return df
        except Exception as e:
            logger.error(f"Error creating health-risk interaction: {str(e)}")
            raise

    def _create_healthcare_access_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create healthcare access index"""
        try:
            df['HealthcareAccessIndex'] = (
                df['AnyHealthcare'] * 2.0 -  # Having healthcare is strongly positive
                df['NoDocbcCost'] * 1.5    # Cost barriers are negative
            ) / 3.5  # Normalize to roughly -1 to 1 scale
            
            logger.info("Created healthcare access index")
            return df
        except Exception as e:
            logger.error(f"Error creating healthcare access index: {str(e)}")
            raise