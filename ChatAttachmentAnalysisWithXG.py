import pandas as pd
import numpy as np
import xgboost as xgb

# for data visualization:
import matplotlib.pyplot as plt
import seaborn as sns


# for regression:
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

datafile_path = "data/chat_transcripts_with_embeddings_and_scores3.csv"

df = pd.read_csv(datafile_path)
df['embedding'] = df['embedding'].apply(lambda x: [float(num) for num in x.strip('[]').split(',')])

y_columns = ['avoide', 'avoida', 'avoidb', 'avoidc', 'avoidd', 'anxietye', 'anxietya', 'anxietyb', 'anxietyc', 'anxietyd']
y = df[y_columns].values

X_train, X_test, y_train, y_test = train_test_split(list(df.embedding.values), y, test_size=0.2, random_state=42)

# Convert y_test to a DataFrame
y_test_df = pd.DataFrame(y_test, columns=y_columns)

# Save the true scores to a CSV file
y_test_df.to_csv("data/true_scoresXG.csv", index=False)


xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
# # Set L1 regularization term (alpha) to 10 and L2 regularization term (lambda) to 2
# xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, lambda = 2, n_estimators = 10)

multioutput_reg = MultiOutputRegressor(xg_reg)

multioutput_reg.fit(np.array(X_train).tolist(), y_train)

preds = multioutput_reg.predict(np.array(X_test).tolist())

# Save the predictions to a CSV
preds_df = pd.DataFrame(preds, columns=['avoide', 'avoida', 'avoidb', 'avoidc', 'avoidd', 'anxietye', 'anxietya', 'anxietyb', 'anxietyc', 'anxietyd'])
preds_df.to_csv("data/predictionsXG.csv", index=False)

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)

print(f"ada-002 embedding performance on chat transcripts: mse={mse:.2f}, mae={mae:.2f}")

# The mean squared error (MSE) and mean absolute error (MAE) are both metrics for assessing the performance of our regression model.

# MSE is calculated by taking the average of the squared differences between the predicted and actual values. It gives more weight to larger errors because they are squared in the calculation. This means that a model could have a relatively high MSE due to a few large errors, even if it made smaller errors on a majority of the instances.

# MAE, on the other hand, is calculated by taking the average of the absolute differences between the predicted and actual values. This metric gives equal weight to all errors and is less sensitive to outliers than MSE.

# Modified visualization code for clarity and interpretability
# Define the column names based on the provided code snippet
column_names = ['avoide', 'avoida', 'avoidb', 'avoidc', 'avoidd', 
                'anxietye', 'anxietya', 'anxietyb', 'anxietyc', 'anxietyd']

# Modified visualization code for clarity and interpretability
# # Create a DataFrame for the actual values
y_test_df = pd.DataFrame(y_test, columns=column_names)
# Create a 2x5 subplot grid
fig, axes = plt.subplots(2, 5, figsize=(25, 12))

# Loop over each column
for idx, col in enumerate(column_names):
    # Plot the actual values
    sns.histplot(y_test_df[col], bins=30, ax=axes[idx//5, idx%5], color='blue', kde=True, label='Actual')
    
    # Plot the predicted values
    sns.histplot(preds_df[col], bins=30, ax=axes[idx//5, idx%5], color='red', kde=True, label='Predicted')
    
    # Set the title of the subplot
    axes[idx//5, idx%5].set_title(f"Distribution of {col}", fontsize=14)
    axes[idx//5, idx%5].legend()

# Adjust the layout
plt.tight_layout()
plt.show()