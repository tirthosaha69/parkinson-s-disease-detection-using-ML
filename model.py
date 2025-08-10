import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load("Model/parkinson_model.pkl")
scaler = joblib.load("Model/scaler.pkl")

def predict_parkinson(input_data):
    # Convert to numpy array
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    # Scale the input
    std_data = scaler.transform(input_data_as_numpy_array)

    # Predict
    prediction = model.predict(std_data)

    return "The Person has Parkinson's Disease" if prediction[0] == 1 else "The Person does not have Parkinson's Disease"
