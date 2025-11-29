import pandas as pd
import numpy as np

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    print("=== HANDLING MISSING VALUES ===")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values > 0]
    
    if missing_columns.empty:
        print("No missing values found in the dataset")
        return df
    
    print(f"Columns with missing values: {list(missing_columns.index)}")
    
    # For numerical columns, fill with median
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    numerical_missing = [col for col in numerical_columns if col in missing_columns.index]
    
    for col in numerical_missing:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        print(f"Filled missing values in '{col}' with median: {median_value}")
    
    # For categorical columns, fill with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    categorical_missing = [col for col in categorical_columns if col in missing_columns.index]
    
    for col in categorical_missing:
        mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
        df[col].fillna(mode_value, inplace=True)
        print(f"Filled missing values in '{col}' with mode: {mode_value}")
    
    print("Missing values handled successfully")
    return df

def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    print("=== REMOVING DUPLICATES ===")
    
    initial_shape = df.shape
    df = df.drop_duplicates()
    final_shape = df.shape
    
    duplicates_removed = initial_shape[0] - final_shape[0]
    print(f"Removed {duplicates_removed} duplicate rows")
    print(f"Dataset shape: {initial_shape} -> {final_shape}")
    
    return df

def convert_data_types(df):
    """
    Convert data types appropriately
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with corrected data types
    """
    print("=== CONVERTING DATA TYPES ===")
    
    # Convert date-related columns to appropriate types
    date_columns = ['arrival_year', 'arrival_month', 'arrival_date']
    existing_date_cols = [col for col in date_columns if col in df.columns]
    
    # Ensure date columns are integers
    for col in existing_date_cols:
        if df[col].dtype != 'int64':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            print(f"Converted '{col}' to integer type")
    
    # Convert categorical columns
    categorical_columns = [
        'type_of_meal_plan', 'room_type_reserved', 'market_segment_type',
        'booking_status'
    ]
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
            print(f"Converted '{col}' to category type")
    
    print("Data types converted successfully")
    return df

def validate_data_ranges(df):
    """
    Validate data ranges and remove invalid rows
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with invalid rows removed
    """
    print("=== VALIDATING DATA RANGES ===")
    
    initial_shape = df.shape
    
    # Validate numerical columns have reasonable values
    validation_rules = {
        'no_of_adults': (0, 20),
        'no_of_children': (0, 20),
        'no_of_weekend_nights': (0, 20),
        'no_of_week_nights': (0, 50),
        'lead_time': (0, 1000),
        'arrival_year': (2017, 2025),
        'arrival_month': (1, 12),
        'arrival_date': (1, 31),
        'no_of_previous_cancellations': (0, 50),
        'no_of_previous_bookings_not_canceled': (0, 50),
        'avg_price_per_room': (0, 10000),
        'no_of_special_requests': (0, 10)
    }
    
    invalid_rows = pd.Series([False] * len(df))
    
    for col, (min_val, max_val) in validation_rules.items():
        if col in df.columns:
            col_invalid = (df[col] < min_val) | (df[col] > max_val)
            invalid_rows |= col_invalid
            invalid_count = col_invalid.sum()
            if invalid_count > 0:
                print(f"Found {invalid_count} invalid values in '{col}' (outside range [{min_val}, {max_val}])")
    
    # Remove invalid rows
    df = df[~invalid_rows]
    
    final_shape = df.shape
    removed_rows = initial_shape[0] - final_shape[0]
    print(f"Removed {removed_rows} rows with invalid values")
    print(f"Dataset shape: {initial_shape} -> {final_shape}")
    
    return df

def clean_data(df):
    """
    Perform complete data cleaning pipeline
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("=== STARTING DATA CLEANING PIPELINE ===")
    
    # Step 1: Handle missing values
    df = handle_missing_values(df)
    
    # Step 2: Remove duplicates
    df = remove_duplicates(df)
    
    # Step 3: Convert data types
    df = convert_data_types(df)
    
    # Step 4: Validate data ranges
    df = validate_data_ranges(df)
    
    print("=== DATA CLEANING COMPLETED ===")
    print(f"Final dataset shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    print("Data cleaning module ready for use")