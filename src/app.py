import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="wide"
)

# Title and description
st.title("🏨 Hotel Booking Cancellation Prediction")
st.markdown("""
This application predicts whether a hotel booking will be canceled based on various features.
The model was trained on historical booking data to help hotels better manage their reservations.
""")

# Load the trained models and encoders
@st.cache_resource
def load_models():
    """Load all trained models and encoders"""
    try:
        # Use absolute path from current directory
        model_dir = os.path.join("..", "models", "final_model")
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            st.warning(f"Model directory not found: {model_dir}")
            return None, None, []
            
        # Load all models
        models = {}
        model_files = {
            "Logistic Regression": "logistic_regression_model.joblib",
            "Random Forest": "random_forest_model.joblib",
            "Gradient Boosting": "gradient_boosting_model.joblib"
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(model_dir, filename)
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                # st.success(f"Loaded {model_name} model")
            else:
                st.warning(f"Model file not found: {model_path}")
        
        # Load encoders
        encoders_path = os.path.join(model_dir, "encoders.joblib")
        if os.path.exists(encoders_path):
            encoders = joblib.load(encoders_path)
        else:
            encoders = None
            st.warning("Encoders file not found")
            
        # Load model metadata to get feature names
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            feature_names = metadata.get('features_used', [])
        else:
            feature_names = []
            st.warning("Model metadata file not found")
            
        return models, encoders, feature_names
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, []

# Load models and feature names
models, encoders, training_features = load_models()

# Sidebar for input parameters
st.sidebar.header(" Booking Parameters")

# Model selection - Radio buttons for better visibility
st.sidebar.subheader("Model Selection")
selected_model = st.sidebar.radio("Choose Model", 
                                  list(models.keys()) if models else ["No models available"],
                                  index=0 if models else 0)

# Create input fields for all features
st.sidebar.subheader("Guest Information")
no_of_adults = st.sidebar.number_input("Number of Adults", min_value=0, max_value=10, value=2, key="adults", step=1)
no_of_children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=0, key="children", step=1)
no_of_weekend_nights = st.sidebar.number_input("Weekend Nights", min_value=0, max_value=10, value=1, key="weekend_nights", step=1)
no_of_week_nights = st.sidebar.number_input("Week Nights", min_value=0, max_value=20, value=2, key="week_nights", step=1)

st.sidebar.subheader("Booking Details")
lead_time = st.sidebar.number_input("Lead Time (days)", min_value=0, max_value=1000, value=50, key="lead_time", step=1)
arrival_year = st.sidebar.number_input("Arrival Year", min_value=2017, max_value=2025, value=2023, key="arrival_year", step=1)
arrival_month = st.sidebar.selectbox("Arrival Month", range(1, 13), index=5, key="arrival_month")
arrival_date = st.sidebar.number_input("Arrival Date", min_value=1, max_value=31, value=15, key="arrival_date", step=1)

st.sidebar.subheader("Room & Services")
avg_price_per_room = st.sidebar.number_input("Average Price per Room", min_value=0.0, value=100.0, key="avg_price", step=10.0, format="%.2f")
no_of_special_requests = st.sidebar.number_input("Special Requests", min_value=0, max_value=5, value=0, key="special_requests", step=1)

st.sidebar.subheader("Guest History")
no_of_previous_cancellations = st.sidebar.number_input("Previous Cancellations", min_value=0, max_value=20, value=0, key="prev_cancellations", step=1)
no_of_previous_bookings_not_canceled = st.sidebar.number_input("Previous Bookings Not Canceled", min_value=0, max_value=20, value=0, key="prev_bookings", step=1)

st.sidebar.subheader("Categorical Features")
required_car_parking_space = st.sidebar.checkbox("Requires Car Parking Space", value=False, key="parking")
repeated_guest = st.sidebar.checkbox("Repeated Guest", value=False, key="repeated")

# Feature engineering functions (same as in training)
def create_features(no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
                   lead_time, avg_price_per_room, no_of_previous_cancellations,
                   no_of_previous_bookings_not_canceled, no_of_special_requests,
                   required_car_parking_space, repeated_guest, arrival_year, 
                   arrival_month, arrival_date):
    """Create engineered features"""
    # Basic features
    total_stay_nights = no_of_weekend_nights + no_of_week_nights
    total_guests = no_of_adults + no_of_children
    adr_per_person = avg_price_per_room / total_guests if total_guests > 0 else avg_price_per_room
    is_weekend_booking = 1 if no_of_weekend_nights > 0 else 0
    
    # Month category (peak season)
    is_peak_season = 1 if arrival_month in [6, 7, 8, 12] else 0
    
    # Previous cancellation rate
    total_previous_bookings = no_of_previous_cancellations + no_of_previous_bookings_not_canceled
    previous_cancellation_rate = no_of_previous_cancellations / total_previous_bookings if total_previous_bookings > 0 else 0
    
    # Special requests per guest
    special_requests_per_guest = no_of_special_requests / total_guests if total_guests > 0 else no_of_special_requests
    
    # Create feature dictionary with only the features used during training
    features = {
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'required_car_parking_space': int(required_car_parking_space),
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'repeated_guest': int(repeated_guest),
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests,
        'total_stay_nights': total_stay_nights,
        'total_guests': total_guests,
        'no_of_babies': 0,  # Default value
        'adr_per_person': adr_per_person,
        'is_weekend_booking': is_weekend_booking,
        'is_peak_season': is_peak_season,
        'previous_cancellation_rate': previous_cancellation_rate,
        'special_requests_per_guest': special_requests_per_guest,
        'Booking_ID_encoded': 0  # Default value
    }
    
    return features

# Prediction button
if st.sidebar.button("Predict Cancellation", type="primary"):
    if models is None or selected_model not in models:
        st.error("Model not loaded. Cannot make predictions.")
    else:
        # Create feature dictionary
        features = create_features(
            no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
            lead_time, avg_price_per_room, no_of_previous_cancellations,
            no_of_previous_bookings_not_canceled, no_of_special_requests,
            required_car_parking_space, repeated_guest, arrival_year, 
            arrival_month, arrival_date
        )
        
        # Only use features that were present during training
        if training_features:
            # Create a new dictionary with only the training features
            filtered_features = {key: features.get(key, 0) for key in training_features}
            feature_df = pd.DataFrame([filtered_features])
        else:
            # Fallback: use all features
            feature_df = pd.DataFrame([features])
        
        # Display input features
        st.subheader("Input Features")
        st.dataframe(feature_df.style.format("{:.4f}"))
        
        # Display prediction for selected model with enhanced precision
        st.subheader(f"Prediction Results - {selected_model}")
        
        try:
            # Get selected model
            model = models[selected_model]
            
            # Make prediction
            prediction = model.predict(feature_df)[0]
            prediction_proba = model.predict_proba(feature_df)[0]
            
            # Get probability of cancellation (class 0 in our encoding)
            cancellation_probability = prediction_proba[0]  # Canceled class
            confirmation_probability = prediction_proba[1]  # Not Canceled class
            
            # Display results with enhanced precision
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Cancellation Probability", f"{cancellation_probability:.4%}")
                
            with col2:
                st.metric("Confirmation Probability", f"{confirmation_probability:.4%}")
                
            with col3:
                confidence = max(cancellation_probability, confirmation_probability)
                st.metric("Model Confidence", f"{confidence:.4%}")
            
            # Show prediction with enhanced detail
            st.subheader("Prediction Outcome")
            if prediction == 0:  # Canceled (based on our encoding)
                st.error("⚠️ Booking is likely to be CANCELLED")
                st.progress(float(cancellation_probability))
                st.caption(f"The model is {cancellation_probability:.2%} confident that this booking will be canceled.")
            else:  # Not Canceled
                st.success("✅ Booking is likely to be CONFIRMED")
                st.progress(float(confirmation_probability))
                st.caption(f"The model is {confirmation_probability:.2%} confident that this booking will be confirmed.")
                
            # Show detailed breakdown
            st.subheader("Detailed Probability Breakdown")
            prob_data = pd.DataFrame({
                'Outcome': ['Canceled', 'Confirmed'],
                'Probability': [cancellation_probability, confirmation_probability]
            })
            st.bar_chart(prob_data.set_index('Outcome'))
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Features being used:", list(feature_df.columns))

# Model information
st.subheader("Model Information")
if models:
    st.info(f"✅ {len(models)} models loaded successfully")
    st.markdown(f"**Selected Model**: {selected_model}")
    st.markdown("""
    **Model Details:**
    - Logistic Regression: Linear model, fast prediction
    - Random Forest: Ensemble of decision trees, good balance of accuracy and speed
    - Gradient Boosting: Sequential tree building, high accuracy
    """)
else:
    st.warning("⚠️ Models not loaded - please train the models first")
    # Show current working directory for debugging
    st.info(f"Current working directory: {os.getcwd()}")
    st.info(f"Models directory expected at: {os.path.join('..', 'models', 'final_model')}")

# Display graphs if available
graphs_dir = os.path.join("..", "graphs")
if os.path.exists(graphs_dir):
    st.subheader("Model Performance Visualizations")
    
    # Get all PNG files in graphs directory
    graph_files = [f for f in os.listdir(graphs_dir) if f.endswith('.png')]
    
    if graph_files:
        # Group graphs by model
        model_graphs = {}
        for file in graph_files:
            # Extract model name from filename
            if 'logistic_regression' in file:
                model_name = 'Logistic Regression'
            elif 'random_forest' in file:
                model_name = 'Random Forest'
            elif 'gradient_boosting' in file:
                model_name = 'Gradient Boosting'
            else:
                model_name = 'Unknown'
                
            if model_name not in model_graphs:
                model_graphs[model_name] = []
            model_graphs[model_name].append(file)
        
        # Display graphs for selected model
        if selected_model in model_graphs:
            st.markdown(f"### {selected_model} Performance")
            model_graph_files = model_graphs[selected_model]
            
            # Display each graph with better formatting
            cols = st.columns(2)  # Create two columns for better layout
            for i, graph_file in enumerate(model_graph_files):
                graph_path = os.path.join(graphs_dir, graph_file)
                if os.path.exists(graph_path):
                    with cols[i % 2]:  # Alternate between columns
                        st.image(graph_path, caption=graph_file.replace('.png', '').replace('_', ' ').title(), use_column_width=True)
    else:
        st.info("No performance graphs available. Run the training pipeline to generate graphs.")
else:
    st.info(f"Graphs directory not found at: {graphs_dir}. Run the training pipeline to generate graphs.")

# Feature importance visualization
st.subheader("Key Factors Affecting Cancellation")
st.markdown("""
Based on the trained models, the most important factors influencing booking cancellations are:
1. **Lead Time** - Longer lead times typically result in higher cancellation rates
2. **Previous Cancellation History** - Guests with prior cancellations are more likely to cancel again
3. **Room Pricing** - Higher room rates may increase cancellation probability
4. **Booking Channel** - Online bookings may have different cancellation patterns
5. **Length of Stay** - Longer stays may be more prone to changes
""")

# How to use
st.subheader("How to Use This Application")
st.markdown("""
1. **Select Model** - Choose from Logistic Regression, Random Forest, or Gradient Boosting
2. **Enter Booking Details** - Fill in the parameters on the left sidebar
3. **Click Predict** - Press the 'Predict Cancellation' button
4. **View Results** - See the probability and prediction for the booking
5. **Compare Models** - Try different models to see how predictions vary
""")

# Footer
st.markdown("---")
st.caption(f"Hotel Booking Cancellation Prediction App | {datetime.now().strftime('%Y-%m-%d')}")