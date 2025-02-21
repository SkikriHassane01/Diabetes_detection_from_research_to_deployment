import os
import json
import shutil # for copying files
from datetime import datetime
from pathlib import Path # for file path manipulation
import pandas as pd # type: ignore
import mlflow # type: ignore
from typing import Optional, Dict

from src.utils.logging_config import setup_logger
logger = setup_logger('data_versioning')

BASE_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__name__))), 'data'))

class DataVersioner:
    """Data versioning system using Mlflow and local Storage"""
    
    def __init__(self, base_dir: str = BASE_DIR, experiment_name: str = "diabetes_classification"):
        """Initialize the versioning system with base directory and Mlflow experiment"""
        
        # set up the local storage directory
        self.base_dir = Path(base_dir)
        self.versions_dir = self.base_dir / "versions"
        os.makedirs(self.versions_dir, exist_ok= True)
        
        # set up the Mlflow experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.info(f"Creating Mlflow experiment: {experiment_name}")
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            logger.info(f"Using existing Mlflow experiment: {experiment_name}")
            self.experiment_id = experiment.experiment_id
        
        # create versions metadata file if it doesn't exist
        self.metadata_file = self.versions_dir / "versions_metadata.json"
        if not self.metadata_file.exists():
            self._save_metadata({})
    
    # TODO: helper function to save the metadata into the json file
    def _save_metadata(self, metadata: Dict) -> None:
        """Save the metadata to the versions metadata file"""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f)
    
    # TODO: load the dict from a json file (used when we want to update the metadata)
    def _load_metadata(self) -> Dict:
        """Load the metadata from the versions metadata file"""
        with open(self.metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata

    # TODO: versioning the dataset
    def version_dataset(self,
                        data: pd.DataFrame,
                        dataset_name: str,
                        dataset_description: Optional[str] = None) -> str:
        """Version the dataset and return MLflow run ID for this version"""
        
        # create a timestamp-based version ID
        version_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        # Create version directory
        version_dir = self.versions_dir / f"{dataset_name}_{version_id}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dataset locally
        dataset_path = version_dir / f"{dataset_name}.csv"
        data.to_csv(dataset_path, index=False)
        logger.info(f"Dataset saved to: {dataset_path}")
        
        # Create metadata
        metadata = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "description": dataset_description,
            "rows": len(data),
            "columns": len(data.columns),
            "local_path": str(dataset_path) # ex : data/versions/diabetes_2021_09_01_12_00_00/diabetes.csv
        }

       # Log to MLflow
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            # Log parameters
            mlflow.log_params({
                "dataset_name": dataset_name,
                "rows": metadata["rows"],
                "columns": metadata["columns"],
                "version_id": version_id
            })
            
            # Log the dataset as an artifact
            mlflow.log_artifact(str(dataset_path))
            
            # Add tags
            mlflow.set_tag("version_time", metadata["timestamp"])
            if dataset_description:
                mlflow.set_tag("description", dataset_description)
            
            # Update local metadata with MLflow run info
            metadata["mlflow_run_id"] = run.info.run_id 
            
            # Update metadata file
            all_metadata = self._load_metadata()
            all_metadata[version_id] = metadata
            self._save_metadata(all_metadata)
            
            return metadata["mlflow_run_id"]
    
    # TODO: return a version of the dataset
    def get_version(self, version_id: str) -> pd.DataFrame:
        """
        Retrieve a specific version of a dataset
        
        Args:
            version_id: Version ID to retrieve
            
        Returns:
            DataFrame containing the dataset version
        """
        metadata = self._load_metadata()
        if version_id not in metadata:
            logger.error(f"No version found with ID: {version_id}")
            raise ValueError(f"No version found with ID: {version_id}")
        
        # Try to get from local storage first
        local_path = metadata[version_id]["local_path"]
        if os.path.exists(local_path):
            logger.info(f"Loading dataset from local path: {local_path}")
            return pd.read_csv(local_path)
        
        # If local file is missing, get from MLflow
        run_id = metadata[version_id]["mlflow_run_id"]
        artifact_uri = mlflow.get_run(run_id).info.artifact_uri
        dataset_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
        
        # Find and load the CSV file
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        if not csv_files:
            logger.error(f"No dataset found in MLflow run: {run_id}")
            raise ValueError(f"No dataset found in MLflow run: {run_id}")
        
        logger.info(f"Loading dataset from MLflow run: {run_id}")
        return pd.read_csv(os.path.join(dataset_path, csv_files[0]))
    
    # TODO: list all the version dataset that we have
    def list_versions(self, dataset_name: Optional[str] = None) -> pd.DataFrame:
        """
        List all versions of a dataset
        
        Args:
            dataset_name: Optional name to filter by
            
        Returns:
            DataFrame containing version information
        """
        metadata = self._load_metadata()
        versions = []
        
        for version_id, info in metadata.items():
            if dataset_name is None or info["dataset_name"] == dataset_name:
                # Get MLflow run info
                run = mlflow.get_run(info["mlflow_run_id"])
                versions.append({
                    "version_id": version_id,
                    "mlflow_run_id": info["mlflow_run_id"],
                    "status": run.info.status,
                    **info
                }) 
        logger.info(f"Found {len(versions)} versions for dataset: {dataset_name}")
        return pd.DataFrame(versions) 

    # TODO: Delete a specific version of the dataset
    def delete_version(self, version_id: str) -> None:
        """
        Delete a specific version both locally and from MLflow
        
        Args:
            version_id: Version ID to delete
        """
        metadata = self._load_metadata()
        if version_id not in metadata:
            logger.error(f"No version found with ID: {version_id}")
            raise ValueError(f"No version found with ID: {version_id}")
        
        # Remove local version directory
        version_path = Path(metadata[version_id]["local_path"]).parent
        if version_path.exists():
            logger.info(f"Deleting version directory: {version_path}")
            shutil.rmtree(version_path)
        
        # Delete MLflow run
        logger.info(f"Deleting MLflow run: {metadata[version_id]['mlflow_run_id']}")
        mlflow.delete_run(metadata[version_id]["mlflow_run_id"])
        
        # Update metadata
        del metadata[version_id]
        self._save_metadata(metadata)  

# if __name__ == "__main__":
#     versioner = DataVersioner()
#     data = pd.read_csv("data\extracted\diabetes_data\diabetes_012_health_indicators_BRFSS2015.csv")
#     version_id = versioner.version_dataset(data, "diabetes")
#     print(version_id)
#     versioner.list_versions("diabetes")
#     versioner.delete_version('2025_02_21_23_49_32')
#     versioner.list_versions("diabetes")   