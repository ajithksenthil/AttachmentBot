# ChatAssessmentAnalysis.py
# Purpose: Script for analyzing chat data using machine learning models, including training, validation, and testing.
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Read your data file
datafile_path = "../data/chat_transcripts_with_embeddings.csv"
df = pd.read_csv(datafile_path)

# Convert embeddings to numpy arrays
df['embedding'] = df['embedding'].apply(lambda x: np.array([float(num) for num in x.strip('[]').split(',')]))
X = np.array(df['embedding'].tolist())

# Process survey responses
import json
from sklearn.preprocessing import MultiLabelBinarizer

def process_survey_responses(survey_responses):
    responses_dict = json.loads(survey_responses.replace("'", "\""))
    flattened_responses = []
    for key, value in responses_dict.items():
        if isinstance(value, list):
            flattened_responses.extend(value)
        else:
            flattened_responses.append(value)
    return flattened_responses

df['processed_survey_responses'] = df['survey_responses'].apply(process_survey_responses)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['processed_survey_responses'])

# Save the MultiLabelBinarizer
mlb_filename = 'models/mlb.pkl'
with open(mlb_filename, 'wb') as file:
    pickle.dump(mlb, file)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a regression model
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.05, max_depth=4, alpha=0, reg_lambda=0.5, n_estimators=200)
multioutput_reg = MultiOutputRegressor(xg_reg)
multioutput_reg.fit(X_train, y_train)

# Validate the model
val_preds = multioutput_reg.predict(X_test)
val_mse = mean_squared_error(y_test, val_preds)
val_mae = mean_absolute_error(y_test, val_preds)
print(f"Validation MSE: {val_mse:.2f}, Validation MAE: {val_mae:.2f}")

# Ensure the 'models' directory exists
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# Save the trained model
model_filename = os.path.join(model_dir, 'trained_model.pkl')
with open(model_filename, 'wb') as file:
    pickle.dump(multioutput_reg, file)
print(f"Model trained and saved as {model_filename}")