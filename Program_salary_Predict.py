import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import locale

# Load Dataset
dataset = pd.read_csv(r"C:\Users\praha\OneDrive\Desktop\Salary_Predict\Salary Data.csv")

# Handling Missing Values
for col in dataset.select_dtypes(include=["float64"]).columns:
    dataset[col].fillna(dataset[col].mean(), inplace=True)

for col in dataset.select_dtypes(include=["object"]).columns:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

# Convert float64 to int64
for col in dataset.select_dtypes(include=["float64"]).columns:
    dataset[col] = dataset[col].astype("int64")

# Encoding Gender
ohe = OneHotEncoder(drop="first")
gender_encoded = ohe.fit_transform(dataset[['Gender']]).toarray()
dataset["Gender_Male"] = gender_encoded[:, 0]
dataset.drop(columns=["Gender"], inplace=True)

# Encoding Education Level
edu_levels = [["PhD", "Master's", "Bachelor's"]]
oe_edu = OrdinalEncoder(categories=edu_levels)
dataset["Education Level"] = oe_edu.fit_transform(dataset[["Education Level"]])

# Grouping Job Titles
def job_title_group(title):
    title = title.lower()
    if any(word in title for word in ['junior', 'assistant', 'entry']):
        return 'Entry-Level'
    elif any(word in title for word in ['manager', 'specialist', 'coordinator', 'supervisor', 'analyst', 'associate']):
        return 'Mid-Level'
    elif any(word in title for word in ['senior', 'director', 'principal', 'lead']):
        return 'Senior-Level'
    elif any(word in title for word in ['vp', 'chief', 'ceo', 'head']):
        return 'Executive-Level'
    else:
        return 'Mid-Level'

dataset['Job_Level'] = dataset['Job Title'].apply(job_title_group)
job_order = [["Entry-Level", "Mid-Level", "Senior-Level", "Executive-Level"]]
oe_job = OrdinalEncoder(categories=job_order)
dataset["Job_Level"] = oe_job.fit_transform(dataset[["Job_Level"]])
dataset.drop(columns=["Job Title"], inplace=True)

# Splitting Data
X = dataset[["Age", "Education Level", "Years of Experience", "Gender_Male", "Job_Level"]]
y = dataset["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=61)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the Model (Updated Version)
joblib.dump(model, r"C:\Users\praha\OneDrive\Desktop\Salary_Predict\Program_salary_Predict.joblib")

# Model Performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R2 Score: {r2}")

# Prediction Plot
plt.scatter(y_test, y_pred, color='blue')
plt.plot(y_test, y_test, color='red')  # Ideal line
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.grid(True)
plt.show()

# Load the Model for Prediction
model = joblib.load(r"C:\Users\praha\OneDrive\Desktop\Salary_Predict\Program_salary_Predict.joblib")

# Predict Salary for New Employee
locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')  # Indian currency format
new_employee = [[30, 4, 5, 1, 2]]  # [Age, Education Level, Experience, Gender_Male, Job_Level]
predicted_salary = model.predict(new_employee)[0]
formatted_salary = locale.currency(predicted_salary, grouping=True, symbol=True)

print(f"Predicted Salary: {formatted_salary}")


import joblib

# Assuming you have trained your model as 'model'
joblib.dump(model, 'Program_salary_Predict.joblib')
print("Model saved successfully!")

