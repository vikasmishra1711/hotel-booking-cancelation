import pandas as pd
import numpy as np

def create_total_stay_feature(df):
    """
    Create total stay nights feature
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with total stay feature
    """
    print("Creating total stay nights feature...")
    
    if 'no_of_weekend_nights' in df.columns and 'no_of_week_nights' in df.columns:
        df['total_stay_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
        print("Created 'total_stay_nights' feature")
    else:
        print("Warning: Required columns for total stay feature not found")
    
    return df

def create_total_guests_feature(df):
    """
    Create total guests feature
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with total guests feature
    """
    print("Creating total guests feature...")
    
    if all(col in df.columns for col in ['no_of_adults', 'no_of_children', 'no_of_babies']):
        df['total_guests'] = df['no_of_adults'] + df['no_of_children'] + df['no_of_babies']
        print("Created 'total_guests' feature")
    elif all(col in df.columns for col in ['no_of_adults', 'no_of_children']):
        df['total_guests'] = df['no_of_adults'] + df['no_of_children']
        print("Created 'total_guests' feature (no babies column found)")
    else:
        print("Warning: Required columns for total guests feature not found")
    
    # Add missing babies column if not present
    if 'no_of_babies' not in df.columns:
        df['no_of_babies'] = 0
        print("Added 'no_of_babies' column with default value 0")
    
    return df

def create_lead_time_category_feature(df):
    """
    Create lead time category feature
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with lead time category feature
    """
    print("Creating lead time category feature...")
    
    if 'lead_time' in df.columns:
        conditions = [
            (df['lead_time'] <= 30),
            (df['lead_time'] > 30) & (df['lead_time'] <= 120),
            (df['lead_time'] > 120)
        ]
        choices = ['Short', 'Medium', 'Long']
        df['lead_time_category'] = np.select(conditions, choices, default='Unknown')
        print("Created 'lead_time_category' feature")
    else:
        print("Warning: 'lead_time' column not found")
    
    return df

def create_adr_per_person_feature(df):
    """
    Create average ADR per person feature
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with ADR per person feature
    """
    print("Creating ADR per person feature...")
    
    if 'avg_price_per_room' in df.columns and 'total_guests' in df.columns:
        # Avoid division by zero
        df['adr_per_person'] = np.where(
            df['total_guests'] > 0,
            df['avg_price_per_room'] / df['total_guests'],
            df['avg_price_per_room']
        )
        print("Created 'adr_per_person' feature")
    else:
        print("Warning: Required columns for ADR per person feature not found")
    
    return df

def create_weekend_booking_flag_feature(df):
    """
    Create weekend booking flag feature
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with weekend booking flag feature
    """
    print("Creating weekend booking flag feature...")
    
    if 'no_of_weekend_nights' in df.columns:
        df['is_weekend_booking'] = (df['no_of_weekend_nights'] > 0).astype(int)
        print("Created 'is_weekend_booking' feature")
    else:
        print("Warning: 'no_of_weekend_nights' column not found")
    
    return df

def create_month_category_feature(df):
    """
    Create month category feature (peak vs non-peak season)
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with month category feature
    """
    print("Creating month category feature...")
    
    if 'arrival_month' in df.columns:
        # Peak season: June, July, August, December (assuming summer and holiday season)
        peak_months = [6, 7, 8, 12]
        df['is_peak_season'] = df['arrival_month'].isin(peak_months).astype(int)
        print("Created 'is_peak_season' feature")
    else:
        print("Warning: 'arrival_month' column not found")
    
    return df

def create_advanced_features(df):
    """
    Create additional advanced features
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with advanced features
    """
    print("Creating advanced features...")
    
    # Total nights feature
    df = create_total_stay_feature(df)
    
    # Total guests feature
    df = create_total_guests_feature(df)
    
    # Lead time category
    df = create_lead_time_category_feature(df)
    
    # ADR per person
    df = create_adr_per_person_feature(df)
    
    # Weekend booking flag
    df = create_weekend_booking_flag_feature(df)
    
    # Month category
    df = create_month_category_feature(df)
    
    # Previous cancellations ratio
    if 'no_of_previous_cancellations' in df.columns and 'no_of_previous_bookings_not_canceled' in df.columns:
        total_previous_bookings = (
            df['no_of_previous_cancellations'] + 
            df['no_of_previous_bookings_not_canceled']
        )
        df['previous_cancellation_rate'] = np.where(
            total_previous_bookings > 0,
            df['no_of_previous_cancellations'] / total_previous_bookings,
            0
        )
        print("Created 'previous_cancellation_rate' feature")
    
    # Special requests per guest
    if 'no_of_special_requests' in df.columns and 'total_guests' in df.columns:
        df['special_requests_per_guest'] = np.where(
            df['total_guests'] > 0,
            df['no_of_special_requests'] / df['total_guests'],
            df['no_of_special_requests']
        )
        print("Created 'special_requests_per_guest' feature")
    
    return df

def engineer_features(df):
    """
    Perform complete feature engineering pipeline
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    print("=== STARTING FEATURE ENGINEERING PIPELINE ===")
    
    # Create basic features (5 minimum required)
    df = create_total_stay_feature(df)
    df = create_total_guests_feature(df)
    df = create_lead_time_category_feature(df)
    df = create_adr_per_person_feature(df)
    df = create_weekend_booking_flag_feature(df)
    
    # Create advanced features
    df = create_advanced_features(df)
    
    print("=== FEATURE ENGINEERING COMPLETED ===")
    print(f"Final dataset shape: {df.shape}")
    print(f"New features created: {len(df.columns) - 17} (approximate)")  # Original columns ~17
    
    return df

if __name__ == "__main__":
    print("Feature engineering module ready for use")