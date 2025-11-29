import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings

def encode_categorical_variables(df):
    """
    Encode categorical variables using appropriate methods
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with encoded categorical variables
    """
    print("=== ENCODING CATEGORICAL VARIABLES ===")
    
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target variable if present
    if 'booking_status' in categorical_columns:
        categorical_columns.remove('booking_status')
    
    print(f"Found {len(categorical_columns)} categorical columns: {categorical_columns}")
    
    # Separate columns by encoding type
    binary_columns = []
    nominal_columns = []
    ordinal_columns = []
    
    for col in categorical_columns:
        unique_values = df[col].nunique()
        if unique_values == 2:
            binary_columns.append(col)
        elif unique_values <= 5:
            nominal_columns.append(col)
        else:
            nominal_columns.append(col)  # Treat as nominal for simplicity
    
    print(f"Binary columns: {binary_columns}")
    print(f"Nominal columns: {nominal_columns}")
    print(f"Ordinal columns: {ordinal_columns}")
    
    # Encode binary columns using Label Encoding
    label_encoders = {}
    for col in binary_columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"Label encoded '{col}' -> '{col}_encoded'")
    
    # Encode nominal columns using One-Hot Encoding
    for col in nominal_columns:
        # Check cardinality to prevent explosion
        if df[col].nunique() <= 10:  # Only one-hot encode if not too many categories
            dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_')
            df = pd.concat([df, dummies], axis=1)
            print(f"One-hot encoded '{col}' ({df[col].nunique()} categories)")
        else:
            # For high cardinality, use label encoding
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            label_encoders[col] = le
            print(f"Label encoded '{col}' (high cardinality: {df[col].nunique()} categories)")
    
    # Handle ordinal columns (if any)
    # For this dataset, we'll treat most as nominal
    
    # Remove original categorical columns
    columns_to_drop = [col for col in categorical_columns if col + '_encoded' in df.columns or col.startswith(tuple(nominal_columns))]
    # But keep the original columns for now, we'll drop them after checking
    
    print(f"Encoding completed. New shape: {df.shape}")
    return df, label_encoders

def prepare_features_for_modeling(df):
    """
    Prepare features for modeling by selecting relevant columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame ready for modeling
    """
    print("=== PREPARING FEATURES FOR MODELING ===")
    
    # Identify columns to keep
    # Keep numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Keep encoded categorical columns (those with '_encoded' or '_')
    encoded_columns = [col for col in df.columns if '_encoded' in col or '_' in col and df[col].dtype in ['uint8', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64']]
    
    # Remove duplicates and ensure we have the right columns
    feature_columns = list(set(numerical_columns + encoded_columns))
    
    # Remove target variable if present in features
    if 'booking_status' in feature_columns:
        feature_columns.remove('booking_status')
    
    # Remove original categorical columns
    original_categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'booking_status' in original_categorical:
        original_categorical.remove('booking_status')
    
    # Keep only the feature columns we want
    final_columns = [col for col in df.columns if col in feature_columns or col == 'booking_status']
    
    df_modeling = df[final_columns].copy()
    
    print(f"Selected {len(final_columns)-1} features for modeling (excluding target)")
    print(f"Final dataset shape: {df_modeling.shape}")
    
    return df_modeling

def encode_target_variable(df):
    """
    Encode the target variable (booking_status)
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with encoded target
        LabelEncoder: Encoder for the target variable
    """
    print("=== ENCODING TARGET VARIABLE ===")
    
    if 'booking_status' in df.columns:
        le_target = LabelEncoder()
        df['booking_status_encoded'] = le_target.fit_transform(df['booking_status'])
        print("Encoded 'booking_status' -> 'booking_status_encoded'")
        print("Class mapping:")
        for i, class_name in enumerate(le_target.classes_):
            print(f"  {i}: {class_name}")
        return df, le_target
    else:
        print("Warning: 'booking_status' column not found")
        return df, None

def handle_categorical_encoding(df):
    """
    Complete categorical encoding pipeline
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with encoded variables
        dict: Dictionary of encoders
    """
    print("=== STARTING CATEGORICAL ENCODING PIPELINE ===")
    
    # Step 1: Encode categorical variables
    df, label_encoders = encode_categorical_variables(df)
    
    # Step 2: Encode target variable
    df, target_encoder = encode_target_variable(df)
    
    # Step 3: Prepare features for modeling
    df = prepare_features_for_modeling(df)
    
    # Combine encoders
    encoders = label_encoders
    if target_encoder:
        encoders['target_encoder'] = target_encoder
    
    print("=== CATEGORICAL ENCODING COMPLETED ===")
    print(f"Final dataset shape: {df.shape}")
    
    return df, encoders

if __name__ == "__main__":
    print("Categorical encoding module ready for use")