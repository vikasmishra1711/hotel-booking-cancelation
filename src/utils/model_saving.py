import joblib
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime

def save_model(model, filepath, model_name="model"):
    """
    Save trained model using joblib
    
    Args:
        model: Trained model to save
        filepath (str): Path to save the model
        model_name (str): Name of the model for logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"=== SAVING {model_name.upper()} ===")
    
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save model
        joblib.dump(model, filepath)
        print(f"Model saved successfully to: {filepath}")
        
        # Also save as pickle for compatibility
        pickle_path = filepath.replace('.joblib', '.pkl') if filepath.endswith('.joblib') else filepath + '.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model also saved as pickle to: {pickle_path}")
        
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def load_model(filepath):
    """
    Load trained model using joblib
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        model: Loaded model or None if failed
    """
    print(f"=== LOADING MODEL FROM {filepath} ===")
    
    try:
        # Try joblib first
        if filepath.endswith('.joblib') or os.path.exists(filepath):
            model = joblib.load(filepath)
            print(f"Model loaded successfully from: {filepath}")
            return model
        else:
            # Try pickle
            pickle_path = filepath.replace('.joblib', '.pkl') if filepath.endswith('.joblib') else filepath + '.pkl'
            with open(pickle_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully from: {pickle_path}")
            return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def save_encoders(encoders, filepath):
    """
    Save encoders using joblib
    
    Args:
        encoders (dict): Dictionary of encoders to save
        filepath (str): Path to save the encoders
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("=== SAVING ENCODERS ===")
    
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save encoders
        joblib.dump(encoders, filepath)
        print(f"Encoders saved successfully to: {filepath}")
        return True
    except Exception as e:
        print(f"Error saving encoders: {str(e)}")
        return False

def load_encoders(filepath):
    """
    Load encoders using joblib
    
    Args:
        filepath (str): Path to the saved encoders
        
    Returns:
        dict: Loaded encoders or None if failed
    """
    print(f"=== LOADING ENCODERS FROM {filepath} ===")
    
    try:
        encoders = joblib.load(filepath)
        print(f"Encoders loaded successfully from: {filepath}")
        return encoders
    except Exception as e:
        print(f"Error loading encoders: {str(e)}")
        return None

def save_preprocessing_pipeline(preprocessing_steps, filepath):
    """
    Save preprocessing pipeline
    
    Args:
        preprocessing_steps (dict): Dictionary of preprocessing steps
        filepath (str): Path to save the pipeline
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("=== SAVING PREPROCESSING PIPELINE ===")
    
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save pipeline
        joblib.dump(preprocessing_steps, filepath)
        print(f"Preprocessing pipeline saved successfully to: {filepath}")
        return True
    except Exception as e:
        print(f"Error saving preprocessing pipeline: {str(e)}")
        return False

def save_model_metadata(model_info, filepath):
    """
    Save model metadata
    
    Args:
        model_info (dict): Dictionary containing model information
        filepath (str): Path to save the metadata
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("=== SAVING MODEL METADATA ===")
    
    try:
        # Add timestamp
        model_info['saved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save as JSON
        import json
        with open(filepath, 'w') as f:
            json.dump(model_info, f, indent=4)
        print(f"Model metadata saved successfully to: {filepath}")
        return True
    except Exception as e:
        print(f"Error saving model metadata: {str(e)}")
        return False

def load_model_metadata(filepath):
    """
    Load model metadata
    
    Args:
        filepath (str): Path to the saved metadata
        
    Returns:
        dict: Loaded metadata or None if failed
    """
    print(f"=== LOADING MODEL METADATA FROM {filepath} ===")
    
    try:
        import json
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        print(f"Model metadata loaded successfully from: {filepath}")
        return metadata
    except Exception as e:
        print(f"Error loading model metadata: {str(e)}")
        return None

def save_complete_model_package(model, encoders, preprocessing_steps, 
                              model_info, save_directory):
    """
    Save complete model package including model, encoders, and metadata
    
    Args:
        model: Trained model
        encoders (dict): Dictionary of encoders
        preprocessing_steps (dict): Preprocessing steps
        model_info (dict): Model information
        save_directory (str): Directory to save all components
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("=== SAVING COMPLETE MODEL PACKAGE ===")
    
    try:
        # Create directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save model
        model_path = os.path.join(save_directory, "model.joblib")
        save_model(model, model_path, "Final Model")
        
        # Save encoders
        encoders_path = os.path.join(save_directory, "encoders.joblib")
        save_encoders(encoders, encoders_path)
        
        # Save preprocessing pipeline
        pipeline_path = os.path.join(save_directory, "preprocessing_pipeline.joblib")
        save_preprocessing_pipeline(preprocessing_steps, pipeline_path)
        
        # Save model metadata
        metadata_path = os.path.join(save_directory, "model_metadata.json")
        save_model_metadata(model_info, metadata_path)
        
        print(f"Complete model package saved to: {save_directory}")
        return True
    except Exception as e:
        print(f"Error saving complete model package: {str(e)}")
        return False

if __name__ == "__main__":
    print("Model saving module ready for use")