from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from src.utils.logging_config import setup_logger

logger = setup_logger('imbalance_handler')

class ImbalanceHandler:
    """
    A simple class for handling imbalanced datasets.
    
    Available strategies:
    - smote: Creates synthetic samples between minority class samples
    - adasyn: Creates more synthetic samples in harder-to-learn regions
    - tomek: Removes majority samples that are too close to minority samples
    - combined: Uses both SMOTE and Tomek links
    """
    
    def __init__(self, strategy='smote', random_state=42):
        """Initialize with chosen resampling strategy."""
        self.strategy = strategy.lower()
        self.random_state = random_state
        
        # Set up the resampling method
        if self.strategy == 'smote':
            self.sampler = SMOTE(random_state=self.random_state)
        elif self.strategy == 'adasyn':
            self.sampler = ADASYN(random_state=self.random_state)
        elif self.strategy == 'tomek':
            self.sampler = TomekLinks()
        elif self.strategy == 'combined':
            self.sampler = SMOTETomek(random_state=self.random_state)
        else:
            raise ValueError("Strategy must be one of: smote, adasyn, tomek, combined")

    def resample(self, X, y):
        """
        Resample the dataset using the selected strategy.
        
        Args:
            X: Features
            y: Target labels
            
        Returns:
            X_resampled, y_resampled: Resampled features and labels
        """
        
        # Apply resampling
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        logger.info(f"Resampled dataset with strategy: {self.strategy}.Previous shape {X.shape} Vs New shape: {X_resampled.shape}")
        
        return X_resampled, y_resampled

    def split_and_resample(self, X, y, test_size=0.2):
        """
        Split the data and apply resampling only to training set.
        
        Args:
            X: Features
            y: Target labels
            test_size: Proportion of dataset to include in test split
            
        Returns:
            X_train_resampled: Resampled training features
            X_test: Test features
            y_train_resampled: Resampled training labels
            y_test: Test labels
        """
        # First split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state
        )
        
        # Then resample only the training data
        X_train_resampled, y_train_resampled = self.resample(X_train, y_train)
        logger.info(f"Split data into train/test sets. Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        logger.info(f"Resampled train set shape: {X_train_resampled.shape}")
        return X_train_resampled, X_test, y_train_resampled, y_test