import logging
import os
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    """
    Function to setup a logger
    
    Args:
        name (str): Name of the logger
        log_file (str): Path to the log file
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    
    return logger

def get_pipeline_logger():
    """
    Get the main pipeline logger
    
    Returns:
        logging.Logger: Main pipeline logger
    """
    log_file = os.path.join("logs", f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")
    return setup_logger("hotel_pipeline", log_file)

def log_step(logger, step_name, message):
    """
    Log a pipeline step
    
    Args:
        logger (logging.Logger): Logger instance
        step_name (str): Name of the step
        message (str): Message to log
    """
    logger.info(f"[{step_name}] {message}")

def log_error(logger, step_name, error):
    """
    Log an error in a pipeline step
    
    Args:
        logger (logging.Logger): Logger instance
        step_name (str): Name of the step
        error (Exception): Error to log
    """
    logger.error(f"[{step_name}] ERROR: {str(error)}")

def log_warning(logger, step_name, warning):
    """
    Log a warning in a pipeline step
    
    Args:
        logger (logging.Logger): Logger instance
        step_name (str): Name of the step
        warning (str): Warning to log
    """
    logger.warning(f"[{step_name}] WARNING: {warning}")

if __name__ == "__main__":
    print("Logger module ready for use")