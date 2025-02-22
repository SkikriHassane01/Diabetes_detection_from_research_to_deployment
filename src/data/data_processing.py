import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from src.utils.logging_config import setup_logger

logger = setup_logger('data_processing')

class DataProcessor:
    """
    Handles all data processing tasks including handling missing values,
    duplicates, outliers, and feature scaling
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DataProcessor with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        # Default configuration
        self.config = config or {
            'health_features': ['MentHlth', 'PhysHlth'],  # Features for outlier treatment and standard scaling
            'bmi_feature': 'BMI',  # Feature for robust scaling
            'target_column': 'Diabetes_012',
            'outlier_threshold': 3.0  # Z-score threshold for outlier detection
        }
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        logger.info("DataProcessor initialized with configuration")
        
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data: DataFrame containing potentially missing values
            
        Returns:
            DataFrame with missing values handled
        """
        # Create a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Check for missing values
        missing_count = df.isnull().sum()
        total_missing = missing_count.sum()
        
        if total_missing > 0:
            logger.info(f"Found {total_missing} missing values across {sum(missing_count > 0)} features")
            
            # For health features and BMI - impute with median
            health_features = self.config['health_features'] + [self.config['bmi_feature']]
            for feature in health_features:
                if feature in df.columns and df[feature].isnull().sum() > 0:
                    median_value = df[feature].median()
                    df[feature].fillna(median_value, inplace=True)
                    logger.info(f"Imputed {feature} missing values with median: {median_value}")
            
            # For all other columns - impute with mode
            other_columns = [col for col in df.columns if col not in health_features]
            for column in other_columns:
                if df[column].isnull().sum() > 0:
                    mode_value = df[column].mode()[0]
                    df[column].fillna(mode_value, inplace=True)
                    logger.info(f"Imputed {column} missing values with mode: {mode_value}")
        else:
            logger.info("No missing values found in the dataset")
            
        return df
    
    def handle_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify and remove duplicate rows from the dataset.
        
        Args:
            data: DataFrame potentially containing duplicates
            
        Returns:
            DataFrame with duplicates removed
        """
        # Check for duplicates
        n_duplicates = data.duplicated().sum()
        
        if n_duplicates > 0:
            logger.info(f"Found {n_duplicates} duplicate rows ({n_duplicates/len(data):.2%} of the dataset)")
            
            # Remove duplicates and reset index
            data_deduped = data.drop_duplicates().reset_index(drop=True)
            logger.info(f"Removed {n_duplicates} duplicate rows, new shape: {data_deduped.shape}")
            return data_deduped
        else:
            logger.info("No duplicate rows found in the dataset")
            return data
    
    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers only in health features (MentHlth and PhysHlth).
        
        Args:
            data: DataFrame potentially containing outliers
            
        Returns:
            DataFrame with outliers handled
        """
        # Create a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Only handle outliers in health features
        for feature in self.config['health_features']:
            if feature not in df.columns:
                continue
                
            # Calculate z-scores for the feature
            z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
            
            # Identify outliers
            outlier_indices = z_scores > self.config['outlier_threshold']
            n_outliers = outlier_indices.sum()
            
            if n_outliers > 0:
                logger.info(f"Found {n_outliers} outliers in {feature}")
                
                # Cap outliers at threshold values (winsorization)
                q1 = df[feature].quantile(0.25)
                q3 = df[feature].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Apply capping
                df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Capped outliers in {feature} to range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return df
    
    def scale_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using different scalers:
        - RobustScaler for BMI
        - StandardScaler for MentHlth and PhysHlth
        
        Args:
            data: DataFrame containing features to scale
            fit: Whether to fit new scalers (True) or use existing ones (False)
            
        Returns:
            DataFrame with scaled features
        """
        # Create a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        try:
            # Scale BMI using RobustScaler
            if self.config['bmi_feature'] in df.columns:
                bmi_values = df[self.config['bmi_feature']].values.reshape(-1, 1)
                if fit:
                    df[self.config['bmi_feature']] = self.robust_scaler.fit_transform(bmi_values)
                else:
                    df[self.config['bmi_feature']] = self.robust_scaler.transform(bmi_values)
                logger.info(f"Applied RobustScaler to {self.config['bmi_feature']}")
            
            # Scale health features using StandardScaler
            health_features = [f for f in self.config['health_features'] if f in df.columns]
            if health_features:
                health_values = df[health_features].values
                if fit:
                    df[health_features] = self.standard_scaler.fit_transform(health_values)
                else:
                    df[health_features] = self.standard_scaler.transform(health_values)
                logger.info(f"Applied StandardScaler to health features: {health_features}")
                
        except Exception as e:
            logger.error(f"Error during feature scaling: {str(e)}")
            raise
            
        return df
    
    def process_data(self, data: pd.DataFrame, fit_scalers: bool = True) -> pd.DataFrame:
        """
        Execute the full data processing pipeline.
        
        Args:
            data: Raw DataFrame to process
            fit_scalers: Whether to fit new scalers or use existing ones
            
        Returns:
            Processed DataFrame ready for model training
        """
        logger.info(f"Starting data processing on data with shape: {data.shape}")
        
        # Apply processing steps in sequence
        data = self.handle_missing_values(data)
        data = self.handle_duplicates(data)
        data = self.handle_outliers(data)
        data = self.scale_features(data, fit=fit_scalers)
        
        # Check for duplicates created during processing
        final_duplicates = data.duplicated().sum()
        if final_duplicates > 0:
            logger.warning(f"Found {final_duplicates} new duplicates after processing, removing them...")
            data = data.drop_duplicates().reset_index(drop=True)
            logger.info(f"Removed new duplicates, final shape: {data.shape}")
        
        logger.info(f"Completed data processing, final shape: {data.shape}")
        
        return data