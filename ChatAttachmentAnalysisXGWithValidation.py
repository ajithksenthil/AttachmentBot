import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read your data file
datafile_path = "data/chat_transcripts_with_embeddings_and_scores.csv"

df = pd.read_csv(datafile_path)

# Convert embeddings to numpy arrays
df['embedding'] = df['embedding'].apply(lambda x: np.array(eval(x)))

# Split the data into features (X) and labels (y)
X = list(df.embedding.values)
y = df[['avoide', 'avoida', 'avoidb', 'avoidc', 'avoidd', 'anxietye', 'anxietya', 'anxietyb', 'anxietyc', 'anxietyd']].values

# Split data into training, validation, and testing sets
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Train your regression model
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
multioutput_reg = MultiOutputRegressor(xg_reg)
multioutput_reg.fit(np.array(X_train).tolist(), y_train)

# Make predictions on the validation data and tune your model parameters accordingly
val_preds = multioutput_reg.predict(np.array(X_val).tolist())
val_mse = mean_squared_error(y_val, val_preds)
val_mae = mean_absolute_error(y_val, val_preds)
print(f"Validation MSE: {val_mse:.2f}, Validation MAE: {val_mae:.2f}")

# After tuning your model, make predictions on the test data
test_preds = multioutput_reg.predict(np.array(X_test).tolist())

# Evaluate your model
test_mse = mean_squared_error(y_test, test_preds)
test_mae = mean_absolute_error(y_test, test_preds)
print(f"Test MSE: {test_mse:.2f}, Test MAE: {test_mae:.2f}")
