# Credit Card Fraud Detection - Streamlit App

A simple Streamlit web application that uses your anomaly detection model to identify fraudulent credit card transactions based on user ID and transaction amount.

## Features

- User-friendly interface for inputting transaction details
- Real-time fraud prediction
- Confidence score for predictions
- Option to upload your joblib model
- Debug tools to inspect model structure

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Place your pre-trained model file (individual_anomaly_models.joblib) in the project directory
   - If you don't have a model, you can generate a dummy model by running:
     ```
     python model_loader.py
     ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Access the application in your web browser at `http://localhost:8501`

3. Enter the transaction details:
   - User ID
   - Transaction Amount

4. Click "Check Transaction" to get the prediction result

## Model Information

The application is designed to work with your anomaly detection model saved as `individual_anomaly_models.joblib`. It assumes:

1. The model is a dictionary of anomaly detection models keyed by user_id
2. Each model in the dictionary is an anomaly detection model (like Isolation Forest)
3. Each model has a `decision_function()` method that returns negative scores for anomalies

The application includes a debug section to help inspect your model structure and adapt the prediction function if needed.

## Project Structure

```
credit-card-fraud-detection/
├── app.py                  # Main Streamlit application
├── model_loader.py         # Utility to create a dummy anomaly model
├── individual_anomaly_models.joblib  # Your pre-trained model
├── requirements.txt        # Required Python packages
└── README.md               # This file
```

## Customization

To adapt this application to your specific model:

1. Modify the `predict_fraud()` function in `app.py` to match your model's expected input format
2. Update the preprocessing logic if needed
3. Add additional input fields if your model requires more features

## License

This project is licensed under the MIT License - see the LICENSE file for details.