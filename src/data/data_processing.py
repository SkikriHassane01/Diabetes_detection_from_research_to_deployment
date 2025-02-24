import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
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
            'outlier_threshold': 3.0,  # Z-score threshold for outlier detection
            'test_size': 0.2,
            'random_state': 42
        }
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        logger.info("DataProcessor initialized with configuration")
        
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df = data.copy()
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
        """Remove duplicate rows from the dataset."""
        n_duplicates = data.duplicated().sum()
        
        if n_duplicates > 0:
            logger.info(f"Found {n_duplicates} duplicate rows ({n_duplicates/len(data):.2%} of the dataset)")
            data_deduped = data.drop_duplicates().reset_index(drop=True)
            logger.info(f"Removed {n_duplicates} duplicate rows, new shape: {data_deduped.shape}")
            return data_deduped
        else:
            logger.info("No duplicate rows found in the dataset")
            return data
    
    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in health features using IQR method."""
        df = data.copy()
        
        for feature in self.config['health_features']:
            if feature not in df.columns:
                continue
                
            z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
            outlier_indices = z_scores > self.config['outlier_threshold']
            n_outliers = outlier_indices.sum()
            
            if n_outliers > 0:
                logger.info(f"Found {n_outliers} outliers in {feature}")
                
                q1 = df[feature].quantile(0.25)
                q3 = df[feature].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Capped outliers in {feature} to range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return df
    
    def split_and_scale_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data and scale features using different scalers:
        - RobustScaler for BMI
        - StandardScaler for MentHlth and PhysHlth
        
        Returns:
            Tuple of (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        try:
            # Split data first
            X = data.drop(columns=[self.config['target_column']])
            y = data[self.config['target_column']]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size= 0.2,
                random_state=42,
                stratify=y
            )
            
            # Scale BMI using RobustScaler
            if self.config['bmi_feature'] in X_train.columns:
                bmi_train = X_train[self.config['bmi_feature']].values.reshape(-1, 1)
                bmi_test = X_test[self.config['bmi_feature']].values.reshape(-1, 1)
                
                X_train[self.config['bmi_feature']] = self.robust_scaler.fit_transform(bmi_train)
                X_test[self.config['bmi_feature']] = self.robust_scaler.transform(bmi_test)
                logger.info(f"Applied RobustScaler to {self.config['bmi_feature']}")
            
            # Scale health features using StandardScaler
            health_features = [f for f in self.config['health_features'] if f in X_train.columns]
            if health_features:
                X_train[health_features] = self.standard_scaler.fit_transform(X_train[health_features])
                X_test[health_features] = self.standard_scaler.transform(X_test[health_features])
                logger.info(f"Applied StandardScaler to health features: {health_features}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error during feature scaling: {str(e)}")
            raise
    
    def process_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Execute the full data processing pipeline.
        
        Returns:
            Tuple of (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        logger.info(f"Starting data processing on data with shape: {data.shape}")
        
        # Apply preprocessing steps
        data = self.handle_missing_values(data)
        data = self.handle_duplicates(data)
        data = self.handle_outliers(data)
        
        # Split and scale the data
        X_train, X_test, y_train, y_test = self.split_and_scale_features(data)
        
        logger.info(f"Completed data processing. Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test