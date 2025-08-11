# random_forsest/model/predict_model.py

import joblib
import numpy as np

# Load the trained model
model = joblib.load('model/random_forest_model.pkl')

def predict_churn(data_dict):
    """
    Predict customer churn from input dictionary
    data_dict = {
        'gender': 1,
        'SeniorCitizen': 0,
        'Partner': 1,
        ...
    }
    """
    # Convert to array
    input_data = np.array(list(data_dict.values())).reshape(1, -1)
    prediction = model.predict(input_data)
    return 'Churn' if prediction[0] == 1 else 'No Churn'
