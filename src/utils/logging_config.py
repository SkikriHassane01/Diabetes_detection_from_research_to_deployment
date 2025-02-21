import os 
import sys
from datetime import datetime
import logging 
from logging.handlers import RotatingFileHandler # prevent unlimited log growth
from typing import Optional

# get the log dir 
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'Logs'
)

def setup_logger(name: str, # ex if name="data_ingestion" => create directory under Logs/data_ingestion/___.log
                log_level: int = logging.INFO,
                log_dir : str = LOG_DIR, # Logs/...
                console_output : bool = True
                ) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name, typically the __name__ form the calling module
        log_level : Logging level (info, debug, error)
        log_dir: where to store the log files 
        console_output: whether to output logs to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    log_dir = os.path.join(log_dir, name)
    # we will configure handlers if they haven't been configured yet to prevents adding duplicate handlers
    if not logger.handlers:
        os.makedirs(log_dir, exist_ok=True)
        logger.setLevel(log_level)
        
        # create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # create file handler for all logs
        log_file = os.path.join(
            log_dir,
            f"{datetime.now().strftime('%Y_%m_%d')}_{name}.log"
        )
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes= 2 * 1024 * 1024, # 2MB
            backupCount= 5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Create error file handler for errors
        log_error_file = os.path.join(
            log_error_file,
            f"{datetime.now().strftime('%Y_%m_%d')}_errors_{name}.log"
        )
        error_file_handler = RotatingFileHandler(
            log_error_file,
            maxBytes= 2 * 1024 * 1024,
            backupCount= 5
        )
        error_file_handler.setLevel(log_level)
        error_file_handler.setFormatter(formatter)
        logger.addHandler(error_file_handler)

        # Create console handler if requested
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    return logger

def get_pipeline_logger(pipeline_name: str,
                       log_level: int = logging.INFO) -> logging.Logger:
    """
    Create a dedicated logger for a specific pipeline run.
    
    Args:
        pipeline_name: Name of the pipeline
        log_level: Logging level
        
    Returns:
        Logger configured for the pipeline run
    """
    pipeline_log_dir = os.path.join(LOG_DIR, "pipelines", pipeline_name)
    return setup_logger(pipeline_name,
                        log_level=log_level,
                        log_dir=pipeline_log_dir)