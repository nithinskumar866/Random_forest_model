# app.py

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and feature names
model = joblib.load('model/churn_model.pkl')
model_features = joblib.load('model/model_features.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form = request.form

    input_dict = {
        'gender': form['gender'],
        'SeniorCitizen': int(form['SeniorCitizen']),
        'Partner': form['Partner'],
        'Dependents': form['Dependents'],
        'tenure': int(form['tenure']),
        'PhoneService': form['PhoneService'],
        'MultipleLines': form['MultipleLines'],
        'InternetService': form['InternetService'],
        'OnlineSecurity': form['OnlineSecurity'],
        'OnlineBackup': form['OnlineBackup'],
        'DeviceProtection': form['DeviceProtection'],
        'TechSupport': form['TechSupport'],
        'StreamingTV': form['StreamingTV'],
        'StreamingMovies': form['StreamingMovies'],
        'Contract': form['Contract'],
        'PaperlessBilling': form['PaperlessBilling'],
        'PaymentMethod': form['PaymentMethod'],
        'MonthlyCharges': float(form['MonthlyCharges']),
        'TotalCharges': float(form['TotalCharges'])
    }

    # Create dataframe
    input_df = pd.DataFrame([input_dict])

    # One-hot encoding
    input_encoded = pd.get_dummies(input_df)

    # Align with model features
    for col in model_features:
        if col not in input_encoded:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_features]

    # Predict
    pred = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1] * 100

    prediction_text = 'Customer will churn.' if pred == 1 else 'Customer will not churn.'
    probability_text = f'Prediction confidence: {prob:.2f}%'

    return render_template('index.html', prediction_text=prediction_text, probability_text=probability_text)

if __name__ == '__main__':
    app.run(debug=True)
