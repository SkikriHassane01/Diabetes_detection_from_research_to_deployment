import pandas as pd # type: ignore
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
                'Diabetes_012', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 
                'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity',
                'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
                'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
                'Sex', 'Age', 'Education', 'Income'
            ],
            'target_column': 'Diabetes_012',
            'expected_classes': [0, 1],
        }
        logger.info("DataValidator initialized with configuration")
    
    # TODO: validate the data
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
    
    # TODO: Validate basic properties
    def _validate_basic_properties(self, data: pd.DataFrame, results: Dict) -> None:
        
        # check data type
        if not isinstance(data, pd.DataFrame):
            results['is_valid'] = False
            results['errors'].append("Data must be a pandas DataFrame")
            return
        
        #check if data is empty
        if data.empty:
            results['is_valid'] = False
            results['errors'].append("Data is empty")
            return
        
        # check the length of the data
        if len(data) == 0:
            results['is_valid'] = False
            results['errors'].append("Data has zero rows")
            return
        
        logger.info(f"Basic validation passed. Data has {len(data)} rows and {len(data.columns)} columns")
    
    # TODO: validate the columns   
    def _validate_columns(self, data: pd.DataFrame, results: Dict) -> None:
        """
        Validate that all expected columns exist in the dataset.
        
        Args:
            data: DataFrame to validate
            results: Dictionary to update with validation results
        """
        missing_columns = [col for col in self.config['expected_columns'] 
                         if col not in data.columns]
                         
        if missing_columns:
            results['is_valid'] = False
            results['errors'].append(f"Missing expected columns: {missing_columns}")
            logger.error(f"Column validation failed. Missing: {missing_columns}")
        else:
            logger.info("Column validation passed. All expected columns present.")
            
        # Check for unexpected columns
        unexpected_columns = [col for col in data.columns 
                            if col not in self.config['expected_columns']]
                            
        if unexpected_columns:
            results['warnings'].append(f"Found unexpected columns: {unexpected_columns}")
            logger.warning(f"Found {len(unexpected_columns)} unexpected columns")
    
    # TODO: Validate the target column
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
            
# if __name__ == '__main__':
#     dv = DataValidator()
#     data = pd.read_csv('data/extracted/diabetes_data/diabetes_012_health_indicators_BRFSS2015.csv')
#     results = dv.validate_data(data)
#     print(results)