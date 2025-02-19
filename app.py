from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('Program_salary_Predict.joblib')

# Function to encode categorical inputs
def preprocess_input(Age, Education_Level, Years_of_Experience, Gender_Male, Job_Level):
    # Updated education mapping to match training data encoding:
    edu_map = {"PhD": 0, "Master's": 1, "Bachelor's": 2}
    Education_Level_num = edu_map.get(Education_Level, -1)
    return np.array([[Age, Education_Level_num, Years_of_Experience, Gender_Male, Job_Level]])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Age = float(request.form['Age'])
    Education_Level = request.form['Education_Level']
    Years_of_Experience = float(request.form['Years_of_Experience'])
    Gender_Male = float(request.form['Gender_Male'])  # 1 for Male, 0 for Female
    Job_Level = float(request.form['Job_Level'])

    input_features = preprocess_input(Age, Education_Level, Years_of_Experience, Gender_Male, Job_Level)
    prediction = model.predict(input_features)[0]

    return render_template('index.html', 
                        prediction_text=f'Predicted Salary: ₹{prediction:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)
