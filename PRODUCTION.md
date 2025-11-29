# Production Deployment Guide

This document provides instructions for loading and using the hotel booking cancellation prediction model in a production environment.

## Loading Models for Production Use

To load and use the trained models in a production environment:

```python
import joblib
import pandas as pd
import os

# Load the trained model (best performing model is Gradient Boosting)
model_path = os.path.join("models", "final_model", "gradient_boosting_model.joblib")
model = joblib.load(model_path)

# Load encoders (if needed for categorical variables)
encoders_path = os.path.join("models", "final_model", "encoders.joblib")
encoders = joblib.load(encoders_path)

# Load model metadata to get feature names
import json
metadata_path = os.path.join("models", "final_model", "model_metadata.json")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
    
feature_names = metadata['features_used']
```

## Making Predictions in Production

```python
# Create a feature dictionary with the same features used during training
features = {
    'no_of_adults': 2,
    'no_of_children': 0,
    'no_of_weekend_nights': 1,
    'no_of_week_nights': 2,
    'required_car_parking_space': 0,
    'lead_time': 50,
    'arrival_year': 2023,
    'arrival_month': 6,
    'arrival_date': 15,
    'repeated_guest': 0,
    'no_of_previous_cancellations': 0,
    'no_of_previous_bookings_not_canceled': 0,
    'avg_price_per_room': 100.0,
    'no_of_special_requests': 0,
    'total_stay_nights': 3,
    'total_guests': 2,
    'no_of_babies': 0,
    'adr_per_person': 50.0,
    'is_weekend_booking': 1,
    'is_peak_season': 1,
    'previous_cancellation_rate': 0.0,
    'special_requests_per_guest': 0.0,
    'Booking_ID_encoded': 0
}

# Create DataFrame with only the features used during training
feature_df = pd.DataFrame([features])
feature_df = feature_df[feature_names]  # Ensure correct order

# Make prediction
prediction = model.predict(feature_df)[0]
prediction_proba = model.predict_proba(feature_df)[0]

# Interpret results
cancellation_probability = prediction_proba[0]  # Probability of cancellation
confirmation_probability = prediction_proba[1]  # Probability of confirmation

if prediction == 0:
    print(f"Booking is likely to be CANCELLED (Probability: {cancellation_probability:.2%})")
else:
    print(f"Booking is likely to be CONFIRMED (Probability: {confirmation_probability:.2%})")
```

## Streamlit Application Deployment

The project uses Streamlit for the web interface. To deploy the Streamlit application in production:

### Running the Streamlit App

```bash
cd src
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Streamlit Production Deployment Options

1. **Streamlit Cloud**: Deploy directly to Streamlit's cloud platform
2. **Docker Container**: Package the application in a Docker container
3. **Cloud Platforms**: Deploy to AWS, Azure, or Google Cloud
4. **On-Premise Servers**: Run on your own servers

## Docker Deployment

To deploy the application using Docker:

1. **Build the Docker image**:
   ```bash
   docker build -t hotel-booking-cancellation .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8501:8501 -v ./models:/app/models hotel-booking-cancellation
   ```

3. **Using Docker Compose**:
   ```bash
   docker-compose up -d
   ```

## Jenkins CI/CD Pipeline

The project includes a Jenkinsfile for continuous integration and deployment:

1. **Setup Jenkins**:
   - Install Jenkins on your server
   - Install required plugins (Docker, Python, Git)
   - Configure Python tool installation

2. **Create Pipeline Job**:
   - Create a new pipeline job in Jenkins
   - Point it to the Jenkinsfile in your repository
   - Configure GitHub webhook for automatic builds

3. **Pipeline Stages**:
   - Checkout code from repository
   - Setup Python environment
   - Run tests
   - Train models
   - Build Docker image
   - Deploy to production

## AWS EC2 Deployment

To deploy the application to AWS EC2:

1. **Using CloudFormation**:
   - Use the provided `aws-cloudformation.yaml` template
   - Launch the stack with your parameters
   - The template creates an EC2 instance with Docker pre-installed

2. **Manual Deployment**:
   - Launch an EC2 instance (Amazon Linux 2 recommended)
   - Install Docker and Docker Compose
   - Copy the application files to the instance
   - Run `docker-compose up -d`

3. **Using Deployment Scripts**:
   - Modify `deploy_to_ec2.sh` or `deploy_to_ec2.bat` with your EC2 details
   - Run the script to build and deploy the application

## Integrating with Existing Systems

To integrate the prediction functionality into existing systems, you can import and use the prediction logic directly:

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from app import create_features, load_models

# Load models
models, encoders, training_features = load_models()

# Select model (e.g., Gradient Boosting - best performing model)
selected_model = models['Gradient Boosting']

# Create features
features = create_features(
    no_of_adults=2,
    no_of_children=0,
    no_of_weekend_nights=1,
    no_of_week_nights=2,
    lead_time=50,
    avg_price_per_room=100.0,
    no_of_previous_cancellations=0,
    no_of_previous_bookings_not_canceled=0,
    no_of_special_requests=0,
    required_car_parking_space=False,
    repeated_guest=False,
    arrival_year=2023,
    arrival_month=6,
    arrival_date=15
)

# Only use features that were present during training
if training_features:
    filtered_features = {key: features.get(key, 0) for key in training_features}
    feature_df = pd.DataFrame([filtered_features])
else:
    feature_df = pd.DataFrame([features])

# Make prediction
prediction = selected_model.predict(feature_df)[0]
prediction_proba = selected_model.predict_proba(feature_df)[0]

cancellation_probability = prediction_proba[0]
confirmation_probability = prediction_proba[1]
```

## Model Monitoring and Retraining

1. **Performance Monitoring**: Track prediction accuracy and model drift over time
2. **Data Drift Detection**: Monitor changes in input data distribution
3. **Automated Retraining**: Use the GitHub Actions workflow or schedule `main.py` to retrain periodically
4. **A/B Testing**: Deploy multiple model versions and compare performance

## Environment Variables for Production

Set these environment variables for production deployment:

```bash
# Model paths
MODEL_PATH=models/final_model/gradient_boosting_model.joblib
ENCODERS_PATH=models/final_model/encoders.joblib
METADATA_PATH=models/final_model/model_metadata.json

# Streamlit settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```