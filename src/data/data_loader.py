import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """
    Load hotel reservation data from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        # Load the data
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def load_and_inspect_data(file_path):
    """
    Load data and perform initial inspection
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and inspected dataframe
    """
    df = load_data(file_path)
    
    if df is not None:
        print("\n=== DATASET INFO ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print("\n=== DATA TYPES ===")
        print(df.dtypes)
        
        print("\n=== MISSING VALUES ===")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found")
        
        print("\n=== DUPLICATES ===")
        duplicates = df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        
        print("\n=== SAMPLE DATA ===")
        print(df.head())
        
        print("\n=== TARGET DISTRIBUTION ===")
        if 'booking_status' in df.columns:
            print(df['booking_status'].value_counts())
    
    return df

if __name__ == "__main__":
    # Test the function
    # Try multiple possible paths
    possible_paths = [
        os.path.join("..", "..", "data", "hotel_reservations.csv"),
        os.path.join("..", "data", "hotel_reservations.csv"),
        os.path.join("data", "hotel_reservations.csv"),
        "hotel_reservations.csv"
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            df = load_and_inspect_data(path)
            break
    
    if df is None:
        print("Error: Could not find hotel_reservations.csv in any expected location")
        print("Please ensure the file exists in the data directory")