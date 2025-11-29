import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def detect_outliers_iqr(df, column):
    """
    Detect outliers using Interquartile Range (IQR) method
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to check for outliers
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def detect_outliers_zscore(df, column, threshold=3):
    """
    Detect outliers using Z-Score method
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to check for outliers
        threshold (float): Z-score threshold (default: 3)
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    return df.index.isin(df[column].dropna().index[z_scores > threshold])

def detect_outliers_isolation_forest(df, columns):
    """
    Detect outliers using Isolation Forest algorithm
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of column names to check for outliers
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    try:
        from sklearn.ensemble import IsolationForest
        
        # Prepare data
        data = df[columns].dropna()
        
        # Fit isolation forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(data)
        
        # Create boolean mask
        outlier_mask = pd.Series(False, index=df.index)
        outlier_mask.loc[data.index] = (outlier_labels == -1)
        
        return outlier_mask
    except ImportError:
        print("Warning: sklearn not available, using IQR method instead")
        # Fallback to IQR method for first column
        return detect_outliers_iqr(df, columns[0]) if columns else pd.Series(False, index=df.index)

def identify_outliers(df, method='iqr'):
    """
    Identify outliers in numerical columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        method (str): Method to use ('iqr', 'zscore', 'isolation_forest')
        
    Returns:
        dict: Dictionary with outlier information for each column
    """
    print("=== IDENTIFYING OUTLIERS ===")
    
    # Select numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target variable if present
    if 'booking_status' in numerical_columns:
        numerical_columns.remove('booking_status')
    
    outlier_info = {}
    
    for column in numerical_columns:
        if method == 'iqr':
            outliers = detect_outliers_iqr(df, column)
        elif method == 'zscore':
            outliers = detect_outliers_zscore(df, column)
        elif method == 'isolation_forest':
            outliers = detect_outliers_isolation_forest(df, [column])
        else:
            print(f"Warning: Unknown method '{method}', using IQR")
            outliers = detect_outliers_iqr(df, column)
        
        outlier_count = outliers.sum()
        outlier_percentage = (outlier_count / len(df)) * 100
        
        outlier_info[column] = {
            'count': outlier_count,
            'percentage': outlier_percentage,
            'indices': df[outliers].index.tolist()
        }
        
        if outlier_count > 0:
            print(f"{column}: {outlier_count} outliers ({outlier_percentage:.2f}%)")
    
    return outlier_info

def cap_outliers_iqr(df, column):
    """
    Cap outliers using IQR method (Winsorization)
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to cap outliers
        
    Returns:
        pd.DataFrame: DataFrame with capped outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    
    return df

def log_transform_outliers(df, column):
    """
    Apply log transformation to handle outliers
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to transform
        
    Returns:
        pd.DataFrame: DataFrame with transformed column
    """
    # Add 1 to handle zeros
    df[f'{column}_log'] = np.log1p(df[column])
    return df

def treat_outliers(df, method='cap'):
    """
    Treat outliers in numerical columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        method (str): Treatment method ('cap', 'log_transform', 'remove')
        
    Returns:
        pd.DataFrame: DataFrame with treated outliers
    """
    print(f"=== TREATING OUTLIERS USING {method.upper()} METHOD ===")
    
    # Select numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target variable if present
    if 'booking_status' in numerical_columns:
        numerical_columns.remove('booking_status')
    
    initial_shape = df.shape
    
    for column in numerical_columns:
        if method == 'cap':
            df = cap_outliers_iqr(df, column)
            print(f"Capped outliers in '{column}' using IQR method")
        elif method == 'log_transform':
            # Only apply log transform to positive columns
            if (df[column] > 0).all():
                df = log_transform_outliers(df, column)
                print(f"Applied log transformation to '{column}'")
            else:
                df = cap_outliers_iqr(df, column)
                print(f"Capped outliers in '{column}' using IQR method (contains zeros/negatives)")
        elif method == 'remove':
            outliers = detect_outliers_iqr(df, column)
            df = df[~outliers]
            print(f"Removed {outliers.sum()} rows with outliers in '{column}'")
    
    if method == 'remove':
        final_shape = df.shape
        removed_rows = initial_shape[0] - final_shape[0]
        print(f"Removed {removed_rows} rows with outliers")
        print(f"Dataset shape: {initial_shape} -> {final_shape}")
    
    return df

def visualize_outliers(df, columns=None, save_path=None):
    """
    Create visualizations for outlier detection
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): Specific columns to visualize (default: all numerical)
        save_path (str): Path to save plots (optional)
    """
    # Select numerical columns
    if columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'booking_status' in numerical_columns:
            numerical_columns.remove('booking_status')
    else:
        numerical_columns = columns
    
    # Limit to first 6 columns for visualization
    numerical_columns = numerical_columns[:6]
    
    if not numerical_columns:
        print("No numerical columns to visualize")
        return
    
    n_cols = min(3, len(numerical_columns))
    n_rows = (len(numerical_columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if len(numerical_columns) > 1 else [axes]
    
    for i, column in enumerate(numerical_columns):
        # Box plot
        sns.boxplot(y=df[column], ax=axes[i])
        axes[i].set_title(f'Box Plot of {column}')
        axes[i].set_ylabel(column)
    
    # Hide empty subplots
    for i in range(len(numerical_columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/outlier_boxplots.png", bbox_inches='tight', dpi=300)
    plt.show()

def handle_outliers(df):
    """
    Complete outlier detection and treatment pipeline
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with handled outliers
    """
    print("=== STARTING OUTLIER HANDLING PIPELINE ===")
    
    # Step 1: Identify outliers
    outlier_info = identify_outliers(df, method='iqr')
    
    # Step 2: Visualize outliers
    print("Creating outlier visualizations...")
    # Note: Visualization would be called from main pipeline with save path
    
    # Step 3: Treat outliers (using capping method to preserve data)
    df = treat_outliers(df, method='cap')
    
    print("=== OUTLIER HANDLING COMPLETED ===")
    return df

if __name__ == "__main__":
    print("Outlier detection module ready for use")