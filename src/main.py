import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import modules
from data.data_loader import load_data, load_and_inspect_data
from preprocessing.eda import perform_eda, visualize_distributions
from preprocessing.data_cleaning import clean_data
from preprocessing.feature_engineering import engineer_features
from preprocessing.outlier_detection import handle_outliers
from preprocessing.encoding import handle_categorical_encoding
from models.class_imbalance import handle_class_imbalance
from models.model_training import train_and_compare_models, split_data, get_all_trained_models
from models.hyperparameter_tuning import quick_hyperparameter_tuning  # Use quick tuning
from models.model_evaluation import comprehensive_model_evaluation
from utils.model_saving import save_complete_model_package

def check_existing_models():
    """Check if models already exist"""
    model_dir = os.path.join("..", "models", "final_model")
    required_files = [
        "logistic_regression_model.joblib",
        "random_forest_model.joblib",
        "gradient_boosting_model.joblib"
    ]
    
    if not os.path.exists(model_dir):
        return False
        
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            return False
            
    return True

def main():
    """
    Main pipeline for hotel booking cancellation prediction
    """
    print("=== HOTEL BOOKING CANCELLATION PREDICTION PIPELINE ===")
    
    # Check if models already exist
    if check_existing_models():
        print("Models already exist. Skipping training to save time.")
        print("If you want to retrain, delete the models directory first.")
        return
    
    print("Starting the complete machine learning pipeline...\n")
    
    # Define paths - try multiple possible locations
    possible_data_paths = [
        os.path.join("..", "data", "hotel_reservations.csv"),
        os.path.join("data", "hotel_reservations.csv"),
        "hotel_reservations.csv"
    ]
    
    data_path = None
    for path in possible_data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print("Error: Could not find hotel_reservations.csv in any expected location")
        print("Please ensure the file exists in the data directory")
        return
    
    save_directory = os.path.join("..", "models", "final_model")
    
    # Create necessary directories
    os.makedirs(os.path.join("..", "data"), exist_ok=True)
    os.makedirs(os.path.join("..", "models"), exist_ok=True)
    os.makedirs(os.path.join("..", "visualizations"), exist_ok=True)
    os.makedirs(os.path.join("..", "graphs"), exist_ok=True)
    
    # Step 1: Load and inspect data
    print("Step 1: Loading and inspecting data...")
    df = load_and_inspect_data(data_path)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Step 2: Exploratory Data Analysis
    print("\nStep 2: Performing Exploratory Data Analysis...")
    eda_results = perform_eda(df)
    
    # Step 3: Data Cleaning
    print("\nStep 3: Cleaning data...")
    df_cleaned = clean_data(df.copy())
    
    # Step 4: Feature Engineering
    print("\nStep 4: Engineering features...")
    df_engineered = engineer_features(df_cleaned.copy())
    
    # Step 5: Outlier Detection and Treatment
    print("\nStep 5: Handling outliers...")
    df_outliers_handled = handle_outliers(df_engineered.copy())
    
    # Step 6: Categorical Encoding
    print("\nStep 6: Encoding categorical variables...")
    df_encoded, encoders = handle_categorical_encoding(df_outliers_handled.copy())
    
    # Step 7: Prepare data for modeling
    print("\nStep 7: Preparing data for modeling...")
    
    # Separate features and target
    if 'booking_status_encoded' in df_encoded.columns:
        X = df_encoded.drop(['booking_status', 'booking_status_encoded'], axis=1)
        y = df_encoded['booking_status_encoded']
        target_encoder = encoders.get('target_encoder')
        
        # Get class names
        if target_encoder:
            class_names = target_encoder.classes_
        else:
            class_names = ['Not_Canceled', 'Canceled']
    else:
        # Fallback if encoding failed
        X = df_encoded.drop(['booking_status'], axis=1)
        y = (df_encoded['booking_status'] == 'Canceled').astype(int)
        class_names = ['Not_Canceled', 'Canceled']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Step 8: Handle Class Imbalance
    print("\nStep 8: Handling class imbalance...")
    X_balanced, y_balanced = handle_class_imbalance(X, y, method='smote')
    
    # Step 9: Split Data
    print("\nStep 9: Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X_balanced, y_balanced)
    
    # Step 10: Train All Models
    print("\nStep 10: Training all models...")
    all_models = get_all_trained_models(X_train, y_train)
    
    # Step 11: Evaluate All Models
    print("\nStep 11: Evaluating all models...")
    model_evaluations = {}
    for model_name, model in all_models.items():
        evaluation_results = comprehensive_model_evaluation(
            model, X_test, y_test, 
            feature_names=X.columns.tolist(),
            model_name=model_name,
            class_names=class_names,
            save_path=os.path.join("..", "visualizations")
        )
        model_evaluations[model_name] = evaluation_results
    
    # Step 12: Hyperparameter Tuning (on best model from initial comparison)
    print("\nStep 12: Comparing models and tuning hyperparameters for best model...")
    model_results = train_and_compare_models(X_train, y_train, X_test, y_test)
    best_model_name = model_results['best_model_name']
    
    # Use quick hyperparameter tuning to reduce model size
    tuned_results = quick_hyperparameter_tuning(
        X_train, y_train, 
        model_name=best_model_name
    )
    
    # Get the tuned model
    tuned_model = tuned_results['best_model']
    
    # Step 13: Comprehensive Model Evaluation for Tuned Model
    print("\nStep 13: Evaluating tuned model...")
    feature_names = X.columns.tolist()
    evaluation_results = comprehensive_model_evaluation(
        tuned_model, X_test, y_test, 
        feature_names=feature_names,
        model_name=f"Tuned {best_model_name}",
        class_names=class_names,
        save_path=os.path.join("..", "visualizations")
    )
    
    # Step 14: Save All Models and Encoders
    print("\nStep 14: Saving all models and encoders...")
    
    # Prepare model information
    model_info = {
        'model_type': best_model_name,
        'model_parameters': tuned_results['best_params'],
        'cv_score': float(tuned_results['best_score']),
        'test_metrics': {
            'accuracy': float(evaluation_results['accuracy']),
            'precision': float(evaluation_results['precision']),
            'recall': float(evaluation_results['recall']),
            'f1_score': float(evaluation_results['f1_score']),
            'roc_auc': float(evaluation_results['roc_auc']) if evaluation_results['roc_auc'] else None
        },
        'features_used': feature_names,
        'n_features': len(feature_names),
        'n_training_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    # Save everything
    success = save_complete_model_package(
        model=tuned_model,
        encoders=encoders,
        preprocessing_steps={
            'feature_names': feature_names,
            'class_names': class_names,
            'all_models': list(all_models.keys())  # Save info about all models
        },
        model_info=model_info,
        save_directory=save_directory
    )
    
    # Also save individual models
    for model_name, model in all_models.items():
        model_save_path = os.path.join(save_directory, f"{model_name.replace(' ', '_').lower()}_model.joblib")
        import joblib
        joblib.dump(model, model_save_path)
        print(f"Saved {model_name} model to: {model_save_path}")
    
    if success:
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        print(f"Best model: {best_model_name}")
        print(f"Test F1-Score: {evaluation_results['f1_score']:.4f}")
        print(f"Model saved to: {save_directory}")
        print(f"Visualizations saved to: ../visualizations/")
        print(f"All models trained and saved successfully")
        print(f"Graphs saved to: ../graphs/")
    else:
        print("\n=== PIPELINE COMPLETED WITH SOME ISSUES ===")
        print("Model training and evaluation completed, but saving may have failed.")

if __name__ == "__main__":
    main()