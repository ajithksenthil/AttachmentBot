# ChatAssessmentAnalysis.py
# Purpose: Script for analyzing chat data using machine learning models, including training, validation, and testing.

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read your data file
datafile_path = "data/chat_transcripts_with_features.csv"  # Update this path as necessary
df = pd.read_csv(datafile_path)

# Convert embeddings to numpy arrays
df['embedding'] = df['embedding'].apply(lambda x: np.array([float(num) for num in x.strip('[]').split(',')]))

# Define features (X) and labels (y) - Adjust column names as per your dataset
X = np.array(df['embedding'].tolist())
y = df[['score1', 'score2', 'score3']].values  # Replace with your actual score columns

# Split data into training, validation, and testing sets
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Train the regression model
# Note: You can replace XGBRegressor with any other regression model as per your requirement.
# For instance, you might use RandomForestRegressor or a neural network model from Keras.
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.05, max_depth=4, alpha=0, lam=0.5, n_estimators=200)
multioutput_reg = MultiOutputRegressor(xg_reg)
multioutput_reg.fit(X_train, y_train)

# Save the trained model
model_filename = 'trained_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(multioutput_reg, file)
print(f"Model trained and saved as {model_filename}")

# Validate the model
# Note: You can use other metrics for validation based on your specific needs.
# For instance, you might consider using precision, recall, F1-score, or ROC-AUC for classification tasks.
val_preds = multioutput_reg.predict(X_val)
val_mse = mean_squared_error(y_val, val_preds)
val_mae = mean_absolute_error(y_val, val_preds)
print(f"Validation MSE: {val_mse:.2f}, Validation MAE: {val_mae:.2f}")

# Test the model
test_preds = multioutput_reg.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds)
test_mae = mean_absolute_error(y_test, test_preds)
print(f"Test MSE: {test_mse:.2f}, Test MAE: {test_mae:.2f}")

# Note to Users:
# - Make sure to adjust the data paths and column names to match your dataset.
# - Feel free to experiment with different machine learning models and parameters to find the best fit for your data.
# - The trained model can be used to make predictions on new chat transcript data.
# - Consider re-training the model periodically with new data to keep it updated and improve its accuracy.
