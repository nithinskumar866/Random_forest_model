# 📊 Telco Customer Churn Prediction

## 📌 Overview
This project predicts whether a telecom customer is likely to **churn** (stop using the service) based on various attributes such as demographics, service usage, and contract details.

The model is built using a **Random Forest Classifier** and is deployed with **Flask** as a web application.  
Users can input customer details through a web form and get **real-time churn predictions** along with prediction confidence.

---

## 🖼 Project Preview
![Project Screenshot](path/to/your/screenshot.png)  
*(Replace `path/to/your/screenshot.png` with the actual image path in your repo)*

---

## 📂 Project Structure
```
Telco-Churn-Prediction/
│
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── model/
│   ├── train_model.py
│   ├── churn_model.pkl
│   └── model_features.pkl
│
├── templates/
│   └── index.html
│
├── app.py
├── README.md
└── requirements.txt
```

---

## 🧠 Model Details
- **Algorithm:** Random Forest Classifier
- **Data Preprocessing:**
  - Dropped `customerID`
  - Converted `TotalCharges` to numeric and handled missing values
  - One-hot encoded categorical variables
- **Target Variable:** `Churn` (Yes = 1, No = 0)

---

## 🚀 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/Telco-Churn-Prediction.git
cd Telco-Churn-Prediction
```

### 2️⃣ Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate       # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Train the model
```bash
cd model
python train_model.py
cd ..
```

### 5️⃣ Run the Flask app
```bash
python app.py
```

### 6️⃣ Access in browser
Go to:
```
http://127.0.0.1:5000/
```

---

## 📋 Usage
- Fill in customer details in the form.
- Click **"Predict Churn"**.
- View:
  - Prediction result (Will churn / Will not churn)
  - Confidence percentage

---

## 📊 Example Input & Output

**Example Input:**
```
Gender: Female
SeniorCitizen: 0
Partner: Yes
Dependents: No
Tenure: 5
Phone Service: Yes
Internet Service: Fiber optic
Contract: Month-to-month
Monthly Charges: 70.35
Total Charges: 351.75
```

**Example Output:**
```
Prediction: Customer will churn.
Confidence: 87.42%
```

---

## 🛠 Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas
- HTML / CSS

---

## 📈 Dataset Source
Telco Customer Churn Dataset - Kaggle

---

## 📜 License
This project is licensed under the MIT License.
