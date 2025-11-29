# Hotel Booking Cancellation Prediction

This project predicts whether a hotel booking will be canceled based on various features using machine learning models. The system helps hotels better manage their reservations by identifying bookings that are likely to be canceled.

## Project Overview

The hotel industry faces significant challenges with booking cancellations, which can lead to revenue loss and operational inefficiencies. This machine learning solution analyzes historical booking data to predict cancellations, enabling hotels to make data-driven decisions.

## Features

- **Multiple ML Models**: Implements Logistic Regression, Random Forest, and Gradient Boosting algorithms
- **Streamlit Web Application**: Interactive dashboard for making predictions
- **Comprehensive Data Pipeline**: Includes data cleaning, feature engineering, and preprocessing
- **Model Evaluation**: Detailed performance metrics and visualizations
- **Automated Retraining**: GitHub Actions workflow for continuous model updates

## Project Structure

```
hotel_booking_cancellation/
├── data/
│   └── hotel_reservations.csv          # Dataset
├── graphs/                             # Model performance visualizations
├── models/
│   └── final_model/                    # Trained models and encoders
├── src/
│   ├── app.py                          # Streamlit web application
│   ├── main.py                         # Main pipeline script
│   ├── data/                           # Data loading modules
│   ├── models/                         # Model training and evaluation
│   ├── preprocessing/                  # Data preprocessing modules
│   └── utils/                          # Utility functions
└── .github/workflows/                  # CI/CD pipeline
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KISHANSINHAA/hotel-booking-cancellation.git
   cd hotel-booking-cancellation
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Train the Models** (if not already trained):
   ```bash
   cd src
   python main.py
   ```

2. **Run the Streamlit App**:
   ```bash
   cd src
   streamlit run app.py
   ```

3. Access the application at `http://localhost:8501`

## Model Performance

The models achieve the following performance metrics on the test set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Gradient Boosting (Best) | ~88.0% | ~88.1% | ~88.0% | ~88.0% | ~95.1% |
| Random Forest | ~87.7% | ~87.9% | ~87.7% | ~87.7% | ~94.6% |
| Logistic Regression | ~75.7% | ~75.7% | ~75.7% | ~75.7% | ~83.1% |

## Usage

1. Select a model from the radio buttons in the sidebar
2. Adjust booking parameters using the input controls
3. Click "Predict Cancellation" to see the results
4. View performance visualizations and feature importance

## Production Deployment

For detailed instructions on loading and using the model in production, see [PRODUCTION.md](PRODUCTION.md).

## Automated Pipeline

The project includes a GitHub Actions workflow that:
- Runs on every push to the main branch
- Executes daily at 2 AM UTC
- Retrains models with the latest data
- Archives trained models as artifacts
- Can be extended for automated deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset sourced from hotel booking records
- Built with scikit-learn, pandas, and Streamlit