import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df):
    """
    Perform exploratory data analysis on the hotel reservation dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing EDA results
    """
    print("=== EXPLORATORY DATA ANALYSIS ===")
    
    # Dataset summary
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of Features: {df.shape[1]}")
    print(f"Number of Records: {df.shape[0]}")
    
    # Data types
    print("\n=== DATA TYPES ===")
    print(df.dtypes.value_counts())
    
    # Missing values
    print("\n=== MISSING VALUES ===")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    print(missing_df if not missing_df.empty else "No missing values found")
    
    # Duplicates
    print(f"\n=== DUPLICATES ===")
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    
    # Target distribution
    if 'booking_status' in df.columns:
        print("\n=== TARGET DISTRIBUTION ===")
        target_dist = df['booking_status'].value_counts()
        target_percent = df['booking_status'].value_counts(normalize=True) * 100
        target_summary = pd.DataFrame({
            'Count': target_dist,
            'Percentage': target_percent
        })
        print(target_summary)
    
    # Numerical features analysis
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'booking_status' in numerical_features:
        numerical_features.remove('booking_status')
    
    print(f"\n=== NUMERICAL FEATURES ({len(numerical_features)}) ===")
    if numerical_features:
        print("Statistical Summary:")
        print(df[numerical_features].describe())
    
    # Categorical features analysis
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    if 'booking_status' in categorical_features:
        categorical_features.remove('booking_status')
    
    print(f"\n=== CATEGORICAL FEATURES ({len(categorical_features)}) ===")
    if categorical_features:
        for feature in categorical_features[:5]:  # Show first 5
            print(f"\n{feature}:")
            print(df[feature].value_counts().head())
        if len(categorical_features) > 5:
            print(f"... and {len(categorical_features) - 5} more categorical features")
    
    # Correlation analysis for numerical features
    print("\n=== CORRELATION ANALYSIS ===")
    if len(numerical_features) > 1:
        correlation_matrix = df[numerical_features].corr()
        print("Top 10 feature correlations:")
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_pairs.append((
                    correlation_matrix.index[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j]
                ))
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for i, (feat1, feat2, corr) in enumerate(corr_pairs[:10]):
            print(f"{i+1}. {feat1} - {feat2}: {corr:.3f}")
    
    return {
        'shape': df.shape,
        'missing_values': missing_df,
        'duplicates': duplicates,
        'target_distribution': target_dist if 'booking_status' in df.columns else None,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features
    }

def visualize_distributions(df, save_path=None):
    """
    Create visualizations for data distributions
    
    Args:
        df (pd.DataFrame): Input dataframe
        save_path (str): Path to save plots (optional)
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig_size = (15, 12)
    
    # Target distribution plot
    if 'booking_status' in df.columns:
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=df, x='booking_status')
        plt.title('Distribution of Booking Status')
        plt.xlabel('Booking Status')
        plt.ylabel('Count')
        
        # Add count labels on bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 10), 
                       textcoords='offset points')
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'booking_status_distribution.png'), 
                       bbox_inches='tight', dpi=300)
        plt.show()
    
    # Numerical features distribution
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'booking_status' in numerical_features:
        numerical_features.remove('booking_status')
    
    if numerical_features:
        n_features = min(len(numerical_features), 12)  # Limit to 12 features
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(numerical_features[:n_features]):
            axes[i].hist(df[feature].dropna(), bins=30, alpha=0.7, color='skyblue')
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'numerical_distributions.png'), 
                       bbox_inches='tight', dpi=300)
        plt.show()

if __name__ == "__main__":
    # This would typically be called from main pipeline
    print("EDA module ready for use")