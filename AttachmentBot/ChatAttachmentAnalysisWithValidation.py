import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read your data file
datafile_path = "data/chat_transcripts_with_embeddings_and_scores.csv"

df = pd.read_csv(datafile_path)

# Convert embeddings to numpy arrays
df['embedding'] = df['embedding'].apply(lambda x: [float(num) for num in x.strip('[]').split(',')])

# Split the data into features (X) and labels (y)
X = list(df.embedding.values)
y = df[['avoide', 'avoida', 'avoidb', 'avoidc', 'avoidd', 'anxietye', 'anxietya', 'anxietyb', 'anxietyc', 'anxietyd']].values

# Split data into training, validation, and testing sets
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Train your regression model
rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(X_train, y_train)

# Make predictions on the validation data and adjust your model parameters accordingly
val_preds = rfr.predict(X_val)
val_mse = mean_squared_error(y_val, val_preds)
val_mae = mean_absolute_error(y_val, val_preds)
print(f"Validation MSE: {val_mse:.2f}, Validation MAE: {val_mae:.2f}")

# After adjusting your model parameters, make predictions on the test data
test_preds = rfr.predict(X_test)

# Evaluate your model
test_mse = mean_squared_error(y_test, test_preds)
test_mae = mean_absolute_error(y_test, test_preds)
print(f"Test MSE: {test_mse:.2f}, Test MAE: {test_mae:.2f}")

# The validation set is used during the model building process to assess how well the model is performing.
# It helps tune the model's hyperparameters, prevent overfitting and select the best performing model.
# A lower Mean Squared Error (MSE) and Mean Absolute Error (MAE) on the validation set indicate a better fit of the model.
# These metrics measure the difference between the predicted and actual values.
# Validation MSE: The average of the squares of the differences between the predicted and actual values in the validation set.
# Validation MAE: The average of the absolute differences between the predicted and actual values in the validation set.

# Once we are confident about our model's parameters and performance, we test it on unseen data - the test set.
# The test set provides the final measure of the model's performance.
# It helps us understand how the model will generalize to new, unseen data.
# A lower Mean Squared Error (MSE) and Mean Absolute Error (MAE) on the test set also indicate a better fit of the model.
# Test MSE: The average of the squares of the differences between the predicted and actual values in the test set.
# Test MAE: The average of the absolute differences between the predicted and actual values in the test set.
# Note that if the model's performance on the test set is significantly worse than on the training set, it may be an indication of overfitting.
