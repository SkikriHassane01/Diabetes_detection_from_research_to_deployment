import pandas as pd
import numpy as np 
from typing import Dict, Optional
from src.utils.logging_config import setup_logger

logger = setup_logger('feature_engineering')

class FeatureEngineer:
    """
    Handles feature engineering for diabetes prediction dataset.
    Creates new features based on medical domain knowledge.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineering configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {
            'age_risk_threshold': 40.0,  # Age above which diabetes risk increases
            'bmi_categories': {
                'Underweight': 18.5,
                'Normal': 24.9,
                'Overweight': 29.9,
                'Obese': float('inf')
            },
            'glucose_risk_threshold': 140.0,  # High blood glucose threshold
            'HbA1c_risk_threshold': 5.7,  # Pre-diabetes threshold
        }
        logger.info("FeatureEngineer initialized with configuration")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering transformation")
        
        # Create copy to avoid modifying original data
        df = data.copy()
        
        # Create features
        df = self._create_bmi_features(df)
        df = self._create_age_related_features(df)
        df = self._create_medical_risk_score(df)
        df = self._create_metabolic_score(df)
        df = self._create_lifestyle_score(df)
        df = self._create_interaction_features(df)
        
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        return df
        
    def _create_bmi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create BMI-related features."""
        # BMI Category
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[-np.inf] + list(self.config['bmi_categories'].values()),
            labels=list(self.config['bmi_categories'].keys())
        )
        
        # Convert to numeric for modeling
        df['bmi_category'] = pd.factorize(df['bmi_category'])[0]
        
        logger.info("Created BMI category features")
        return df
        
    def _create_age_related_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age-related features."""
        # Age risk factor (increases after threshold)
        df['age_risk'] = (df['age'] > self.config['age_risk_threshold']).astype(int)
        
        # Age-BMI interaction
        df['age_bmi_interaction'] = df['age'] * df['bmi'] / 100.0
        
        logger.info("Created age-related features")
        return df
        
    def _create_medical_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite medical risk score."""
        df['medical_risk_score'] = (
            df['hypertension'] * 2.0 +  # High impact
            df['heart_disease'] * 2.0 +  # High impact
            df['age_risk'] * 1.5 +      # Medium impact
            (df['bmi_category'] >= 2).astype(int)  # Impact of overweight/obese
        ) / 6.5  # Normalize to 0-1 range
        
        logger.info("Created medical risk score")
        return df
        
    def _create_metabolic_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create metabolic health score."""
        # High glucose risk
        glucose_risk = (df['blood_glucose_level'] > self.config['glucose_risk_threshold']).astype(float)
        
        # High HbA1c risk
        hba1c_risk = (df['HbA1c_level'] > self.config['HbA1c_risk_threshold']).astype(float)
        
        # Combined metabolic score
        df['metabolic_score'] = (
            glucose_risk * 2.0 +  # High impact
            hba1c_risk * 2.0 +   # High impact
            (df['bmi_category'] >= 2).astype(float)  # Impact of overweight/obese
        ) / 5.0  # Normalize to 0-1 range
        
        logger.info("Created metabolic score")
        return df
        
    def _create_lifestyle_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lifestyle risk score based on smoking history and BMI."""
        # Encode smoking history risk
        smoking_risk = pd.get_dummies(df['smoking_history'], prefix='smoking')
        
        # Higher risk for current and former smokers
        risk_weights = {
            'smoking_current': 1.0,
            'smoking_former': 0.7,
            'smoking_ever': 0.7,
            'smoking_not current': 0.5,
            'smoking_never': 0.0,
            'smoking_No Info': 0.5
        }
        
        # Calculate weighted smoking risk
        df['smoking_risk'] = 0
        for col, weight in risk_weights.items():
            if col in smoking_risk.columns:
                df['smoking_risk'] += smoking_risk[col] * weight
        
        # Combine with BMI risk for overall lifestyle score
        df['lifestyle_score'] = (
            df['smoking_risk'] * 0.6 +  # Smoking impact
            (df['bmi_category'] >= 2).astype(float) * 0.4  # BMI impact
        )
        
        logger.info("Created lifestyle score")
        return df
        
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables."""
        # Age-medical interactions
        df['age_hypertension'] = df['age'] * df['hypertension']
        df['age_heart_disease'] = df['age'] * df['heart_disease']
        
        # Medical condition interactions
        df['cardio_metabolic_risk'] = df['hypertension'] * df['heart_disease'] * df['metabolic_score']
        
        # Risk score interactions
        df['combined_risk_score'] = (
            df['medical_risk_score'] * 0.4 +
            df['metabolic_score'] * 0.4 +
            df['lifestyle_score'] * 0.2
        )
        
        logger.info("Created interaction features")
        return df