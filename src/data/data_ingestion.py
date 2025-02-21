import os 
import pandas as pd 
import zipfile
from typing import Optional, Dict
from src.utils.logging_config import setup_logger
from pathlib import Path

logger = setup_logger('data_ingestion')

DATA_PATH = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__name__))), 'data'))

# TODO: Utility class for reading the data
class DataReader:
    readers = {
        '.csv' : pd.read_csv,
        '.xlsx' : pd.read_excel,
        '.xls' : pd.read_excel
    }
    
    @classmethod
    def read_file(cls, data_file:str) -> pd.DataFrame:
        ext = os.path.splitext(data_file)[1].lower()
        reader = cls.readers.get(ext)
        if not reader:
            logger.info(f"Unsupported file format: {ext}")
            raise ValueError(f"Unsupported file format: {ext}")
        logger.info(f"The {ext} Data was loaded successfully")
        return reader(data_file)
    
class DataIngestion:
    """
        Handles loading raw data from various sources (zip, csv, excel).
    """
    def __init__(self, config: Optional[Dict[str, str]] = None):
        self.config = config or {
            "raw_data_path" : DATA_PATH / 'raw',
            "extracted_data_path" : DATA_PATH / 'extracted',
            "supported_format" : ['.zip', '.csv', '.xls', '.xlsx']
        }

        # ensure that the directories exist if not make them
        os.makedirs(self.config['raw_data_path'], exist_ok=True)
        os.makedirs(self.config['extracted_data_path'], exist_ok=True)
        logger.info(f"DataIngestion initialized with config: {self.config}")
    
    # TODO: extracted zip file  
    def extracted_zip_file(self, zip_path: str) -> str:
        """
        Extracted the csv or excel files from the raw data to extracted data directory
        
        Args:
            zip_path: Path to the zip file
        Returns: 
            Path to the directory where files were extracted
        """
        extracted_dir = os.path.join(
            self.config['extracted_data_path'],
            os.path.basename(zip_path).replace(".zip","")
        ) #ex: data/extracted/diabetes
        os.makedirs(extracted_dir,exist_ok=True)
        
        try:
            # extract all the files inside the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
            
            extracted_files = os.listdir(extracted_dir)
            logger.info(f"We extract ==> {len(extracted_files)} files from ==> {zip_path} to ==> {extracted_dir}")
            
            return extracted_dir
                            
        except zipfile.BadZipFile:
            logger.error(f"Failed to extract the zip file in this directory ==> {zip_path}, Not a valid zip file")
            raise ValueError(f"{zip_path} is not a valid zip file")
        except Exception as e:
            logger.error(f"Error extracting the zip file in the ==> {zip_path} directory :: {str(e)}")
            raise ValueError(f"Error extracting the zip file in the ==> {zip_path} directory :: {str(e)}")

    # TODO: Find data file
    def find_data_file(self, directory: str) -> Optional[str]:
        """
        Find the first CSV or Excel file in the given directory.
        
        Args:
            directory: Directory to search in
            
        Returns:
            Path to the first found data file, or None if no suitable file exists
        """
        if not os.path.exists(directory):
            logger.error(f"Directory does not exist: {directory}")
            return None
            
        # Look for CSV files first, then Excel
        file_extensions = ['.csv', '.xlsx', '.xls']
        for ext in file_extensions:
            for file in os.listdir(directory):
                if file.lower().endswith(ext):
                    file_path = os.path.join(directory, file)
                    logger.info(f"Found data file: {file_path}")
                    return file_path
        logger.error(f"No data files found in {directory}")
        return None
    
    # TODO: Reading the data
    def read_data(self, data_file: str) -> pd.DataFrame:
        """
        Read a CSV or Excel file into a DataFrame.
        
        Args:
            data_file: Path to the data file
        Returns:
            DataFrame containing the data
        """
        if data_file is None:
            logger.error(f"No data file existed in the directory {data_file}to read")
            return None
        try:
            data = DataReader.read_file(data_file)
            return data
        except Exception as e:
            logger.error(f"Error reading {data_file}: {str(e)}")
            return None   
        
# if __name__ == "__main__":
    # di = DataIngestion()
    # zip_path = "data/raw/diabetes_data.zip"
    # extracted_dir = di.extracted_zip_file(zip_path)
    # file_path = di.find_data_file(extracted_dir)
    # data = di.read_data(file_path)
    
    
    