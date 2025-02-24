import pandas as pd
from typing import Dict, Any, Optional
from src.utils.logging_config import setup_logger

logger = setup_logger('data_validation')

class DataValidator:
    """
    Handles all data validation logic, ensuring data quality and integrity.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DataValidator with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        # Default configuration
        self.config = config or {
            'expected_columns': [
                'gender', 'age', 'hypertension', 'heart_disease',
                'smoking_history', 'bmi', 'HbA1c_level',
                'blood_glucose_level', 'diabetes'
            ],
            'target_column': 'diabetes',
            'expected_classes': [0, 1],
            'categorical_columns': ['gender', 'smoking_history'],
            'binary_columns': ['hypertension', 'heart_disease', 'diabetes'],
            'continuous_columns': ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'],
            'valid_genders': ['Male', 'Female', 'Other'],
            'valid_smoking_history': ['never', 'current', 'former', 'not current', 'ever', 'No Info']
        }
        logger.info("DataValidator initialized with configuration")
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive validation on the dataset.
        
        Args:
            data: Pandas DataFrame to validate
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        self._validate_basic_properties(data, results)
        if not results['is_valid']:
            logger.error("Basic validation failed. Skipping further validation.")
            return results
            
        self._validate_columns(data, results)
        self._validate_target(data, results)
        if results['is_valid']:
            logger.info("Data validation passed")
        else:
            logger.error(f"Validation generated {len(results['errors'])} errors and {len(results['warnings'])} warnings")
        
        return results
    
    def _validate_basic_properties(self, data: pd.DataFrame, results: Dict) -> None:
        """
        Validate basic properties of the dataset.
        
        Args:
            data: DataFrame to validate
            results: Dictionary to update with validation results
        """
        if not isinstance(data, pd.DataFrame):
            results['is_valid'] = False
            results['errors'].append("Data must be a pandas DataFrame")
            return
        
        if data.empty:
            results['is_valid'] = False
            results['errors'].append("Data is empty")
            return
        
        if len(data) == 0:
            results['is_valid'] = False
            results['errors'].append("Data has zero rows")
            return
        
        logger.info(f"Basic validation passed. Data has {len(data)} rows and {len(data.columns)} columns")
    
    def _validate_columns(self, data: pd.DataFrame, results: Dict) -> None:
        """
        Validate that all expected columns exist and have correct data types.
        
        Args:
            data: DataFrame to validate
            results: Dictionary to update with validation results
        """
        # Check for missing columns
        missing_columns = [col for col in self.config['expected_columns'] 
                         if col not in data.columns]
        if missing_columns:
            results['is_valid'] = False
            results['errors'].append(f"Missing expected columns: {missing_columns}")
            logger.error(f"Column validation failed. Missing: {missing_columns}")
            return
            
        # Check for unexpected columns
        unexpected_columns = [col for col in data.columns 
                            if col not in self.config['expected_columns']]
        if unexpected_columns:
            results['warnings'].append(f"Found unexpected columns: {unexpected_columns}")
            logger.warning(f"Found {len(unexpected_columns)} unexpected columns")
        
        # Validate categorical values
        for col in self.config['categorical_columns']:
            if col == 'gender':
                invalid_values = set(data[col].unique()) - set(self.config['valid_genders'])
                if invalid_values:
                    results['warnings'].append(f"Invalid gender values found: {invalid_values}")
                    
            if col == 'smoking_history':
                invalid_values = set(data[col].unique()) - set(self.config['valid_smoking_history'])
                if invalid_values:
                    results['warnings'].append(f"Invalid smoking_history values found: {invalid_values}")
        
        # Validate binary columns
        for col in self.config['binary_columns']:
            invalid_values = set(data[col].unique()) - {0, 1}
            if invalid_values:
                results['errors'].append(f"Invalid values in binary column {col}: {invalid_values}")
                results['is_valid'] = False
        
        # Validate continuous columns
        for col in self.config['continuous_columns']:
            if not pd.api.types.is_numeric_dtype(data[col]):
                results['errors'].append(f"Column {col} should be numeric")
                results['is_valid'] = False
            elif data[col].min() < 0:
                results['warnings'].append(f"Found negative values in {col}")
        
        if results['is_valid']:
            logger.info("Column validation passed")
    
    def _validate_target(self, data: pd.DataFrame, results: Dict) -> None:
        """
        Validate the target column and its classes.
        
        Args:
            data: DataFrame to validate
            results: Dictionary to update with validation results
        """
        target_col = self.config['target_column']
        
        # Check target column exists
        if target_col not in data.columns:
            results['is_valid'] = False
            results['errors'].append(f"Target column '{target_col}' not found")
            logger.error(f"Target column '{target_col}' not found in dataset")
            return
            
        # Check target classes
        unique_classes = data[target_col].unique()
        missing_classes = [cls for cls in self.config['expected_classes'] 
                         if cls not in unique_classes]
        if missing_classes:
            results['warnings'].append(
                f"Target column missing expected classes: {missing_classes}"
            )
            logger.warning(f"Missing expected classes in target: {missing_classes}")
            
        # Check for unexpected classes
        unexpected_classes = [cls for cls in unique_classes 
                            if cls not in self.config['expected_classes']]
        if unexpected_classes:
            results['errors'].append(
                f"Found unexpected classes in target: {unexpected_classes}"
            )
            results['is_valid'] = False
            logger.error(f"Found unexpected classes in target: {unexpected_classes}")
        else:
            logger.info("Target validation passed")
            
if __name__ == '__main__':
    dv = DataValidator()
    data_path = 'data/extracted/diabetes_prediction_dataset/diabetes_prediction_dataset.csv'
    data = pd.read_csv(data_path)
    results = dv.validate_data(data)
    print(results)