import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
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
            'continuous_features': ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'],
            'categorical_features': ['gender', 'smoking_history'],
            'binary_features': ['hypertension', 'heart_disease'],
            'target_column': 'diabetes',
            'categorical_encoding' : 'label',  # Encoding strategy for categorical features
            'outlier_threshold': 3.0  # Z-score threshold for outlier detection
        }
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.label_encoders = {}
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.encoded_feature_names = []
        logger.info("DataProcessor initialized with configuration")
        
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        """
        df = data.copy()
        
        # Check for missing values
        missing_count = df.isnull().sum()
        total_missing = missing_count.sum()
        
        if total_missing > 0:
            logger.info(f"Found {total_missing} missing values across {sum(missing_count > 0)} features")
            
            # Handle continuous features - impute with median
            for feature in self.config['continuous_features']:
                if feature in df.columns and df[feature].isnull().sum() > 0:
                    median_value = df[feature].median()
                    df[feature].fillna(median_value, inplace=True)
                    logger.info(f"Imputed {feature} missing values with median: {median_value}")
            
            # Handle categorical features - impute with mode
            for feature in self.config['categorical_features']:
                if feature in df.columns and df[feature].isnull().sum() > 0:
                    mode_value = df[feature].mode()[0]
                    df[feature].fillna(mode_value, inplace=True)
                    logger.info(f"Imputed {feature} missing values with mode: {mode_value}")
                    
            # Handle binary features - impute with mode
            for feature in self.config['binary_features']:
                if feature in df.columns and df[feature].isnull().sum() > 0:
                    mode_value = df[feature].mode()[0]
                    df[feature].fillna(mode_value, inplace=True)
                    logger.info(f"Imputed {feature} missing values with mode: {mode_value}")
        else:
            logger.info("No missing values found in the dataset")
            
        return df
    
    def handle_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify and remove duplicate rows from the dataset.
        """
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
        """
        Detect and handle outliers in continuous features.
        """
        df = data.copy()
        
        for feature in self.config['continuous_features']:
            # Calculate z-scores
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
        Scale features using appropriate scalers:
        - RobustScaler for age, blood_glucose_level (more robust to outliers)
        - StandardScaler for bmi, HbA1c_level
        """
        df = data.copy()
        
        try:
            # Scale with RobustScaler
            robust_features = ['age', 'blood_glucose_level']
            robust_features = [f for f in robust_features if f in df.columns]
            if robust_features:
                if fit:
                    df[robust_features] = self.robust_scaler.fit_transform(df[robust_features])
                else:
                    df[robust_features] = self.robust_scaler.transform(df[robust_features])
                logger.info(f"Applied RobustScaler to {robust_features}")
            
            # Scale with StandardScaler
            standard_features = ['bmi', 'HbA1c_level']
            standard_features = [f for f in standard_features if f in df.columns]
            if standard_features:
                if fit:
                    df[standard_features] = self.standard_scaler.fit_transform(df[standard_features])
                else:
                    df[standard_features] = self.standard_scaler.transform(df[standard_features])
                logger.info(f"Applied StandardScaler to {standard_features}")
                
        except Exception as e:
            logger.error(f"Error during feature scaling: {str(e)}")
            raise
            
        return df

    def encode_categorical_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using specified strategy.
        
        Args:
            data: Input DataFrame
            fit: Whether to fit encoders (True for training, False for test)
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = data.copy()
        
        # Get categorical features present in the data
        cat_features = [f for f in self.config['categorical_features'] if f in df.columns]
        
        if not cat_features:
            return df
            
        try:
            if self.config['categorical_encoding'] == 'label':
                # Label encoding
                for feature in cat_features:
                    if fit:
                        # Initialize and fit encoder if not exists
                        if feature not in self.label_encoders:
                            self.label_encoders[feature] = LabelEncoder()
                        # Fit the encoder
                        self.label_encoders[feature].fit(df[feature].astype(str))
                    
                    # Transform the data
                    df[feature] = self.label_encoders[feature].transform(df[feature].astype(str))
                    logger.info(f"Applied label encoding to {feature}")
                    
            elif self.config['categorical_encoding'] == 'onehot':
                # Get categorical data
                categorical_data = df[cat_features]
                
                if fit:
                    # Fit and transform
                    encoded_array = self.onehot_encoder.fit_transform(categorical_data)
                    
                    # Generate feature names
                    self.encoded_feature_names = []
                    for i, feature in enumerate(cat_features):
                        feature_vals = self.onehot_encoder.categories_[i]
                        self.encoded_feature_names.extend([f"{feature}_{val}" for val in feature_vals])
                else:
                    # Transform only
                    encoded_array = self.onehot_encoder.transform(categorical_data)
                
                # Create DataFrame with encoded data
                encoded_df = pd.DataFrame(
                    encoded_array,
                    columns=self.encoded_feature_names,
                    index=df.index
                )
                
                # Drop original categorical columns and join encoded ones
                df = df.drop(columns=cat_features)
                df = pd.concat([df, encoded_df], axis=1)
                logger.info(f"Applied one-hot encoding to {cat_features}")
                
            else:
                raise ValueError(f"Unsupported encoding strategy: {self.config['categorical_encoding']}")
                
        except Exception as e:
            logger.error(f"Error during categorical encoding: {str(e)}")
            raise
            
        return df

    def process_data(self, data: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Execute the full data processing pipeline with consistent shapes.
        
        Args:
            data: Input DataFrame
            test_size: Proportion of data for test set
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Starting data processing on data with shape: {data.shape}")
        
        df = data.copy()
        
        df = self.handle_missing_values(df)
        df = self.handle_duplicates(df)
        
        # Extract target 
        y = df[self.config['target_column']]
        X = df.drop(columns=[self.config['target_column']])
        
        # Split the cleaned data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Process training data
        X_train = self.handle_outliers(X_train)
        X_train = self.encode_categorical_features(X_train, fit=True)
        X_train = self.scale_features(X_train, fit=True)
        
        # Process test data
        X_test = self.handle_outliers(X_test)
        X_test = self.encode_categorical_features(X_test, fit=False)
        X_test = self.scale_features(X_test, fit=False)
        
        # Verify shapes match
        assert len(X_train) == len(y_train), "Training data and labels must have same length"
        assert len(X_test) == len(y_test), "Test data and labels must have same length"
        
        logger.info(f"Completed data processing.")
        logger.info(f"Training shapes - X: {X_train.shape}, y: {y_train.shape}")
        logger.info(f"Test shapes - X: {X_test.shape}, y: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test