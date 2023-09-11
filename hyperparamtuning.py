# import pandas as pd
# import numpy as np

# # for data visualization:
# import matplotlib.pyplot as plt
# import seaborn as sns

# # for regression and hyperparameter tuning:
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# # Read your data file
# datafile_path = "data/chat_transcripts_with_embeddings_and_scores3.csv"

# df = pd.read_csv(datafile_path)

# # Convert embeddings to numpy arrays
# df['embedding'] = df['embedding'].apply(lambda x: [float(num) for num in x.strip('[]').split(',')])

# # Split the data into features (X) and labels (y)
# X = list(df.embedding.values)
# y = df[['avoide', 'avoida', 'avoidb', 'avoidc', 'avoidd', 'anxietye', 'anxietya', 'anxietyb', 'anxietyc', 'anxietyd']].values

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define the hyperparameters and their possible values for Grid Search
# param_grid = {
#     'n_estimators': [10, 50, 100, 150],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt']
# }

# # Initialize the model
# rfr = RandomForestRegressor()

# # Grid search
# grid_search = GridSearchCV(rfr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)

# # Best parameters from grid search
# best_params = grid_search.best_params_

# print(best_params)

# {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}




import pandas as pd
import numpy as np

# for data visualization:
import matplotlib.pyplot as plt
import seaborn as sns

# for regression and hyperparameter tuning:
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read your data file
datafile_path = "data/chat_transcripts_with_embeddings_and_scores3.csv"

df = pd.read_csv(datafile_path)

# Convert embeddings to numpy arrays
df['embedding'] = df['embedding'].apply(lambda x: [float(num) for num in x.strip('[]').split(',')])

# Split the data into features (X) and labels (y)
X = list(df.embedding.values)
y = df[['avoide', 'avoida', 'avoidb', 'avoidc', 'avoidd', 'anxietye', 'anxietya', 'anxietyb', 'anxietyc', 'anxietyd']].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameters and their distributions for Random Search
param_dist = {
    'n_estimators': [10, 50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize the model
rfr = RandomForestRegressor()

# Random search
random_search = RandomizedSearchCV(rfr, param_distributions=param_dist, n_iter=100, cv=5, 
                                   scoring='neg_mean_squared_error', n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

# Best parameters from random search
best_params = random_search.best_params_

print(best_params)
# Train your regression model with the best parameters
rfr_best = RandomForestRegressor(**best_params)
rfr_best.fit(X_train, y_train)

# Make predictions on the test data
preds = rfr_best.predict(X_test)

# Save the predictions to a CSV
preds_df = pd.DataFrame(preds, columns=['avoide', 'avoida', 'avoidb', 'avoidc', 'avoidd', 'anxietye', 'anxietya', 'anxietyb', 'anxietyc', 'anxietyd'])
preds_df.to_csv("data/predictions.csv", index=False)

# Evaluate your model
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)

print(f"Chat transcript embeddings performance with Random Search: mse={mse:.2f}, mae={mae:.2f}")










# # Train your regression model with the best parameters
# rfr_best = RandomForestRegressor(**best_params)
# rfr_best.fit(X_train, y_train)

# # Make predictions on the test data
# preds = rfr_best.predict(X_test)

# # Save the predictions to a CSV
# preds_df = pd.DataFrame(preds, columns=['avoide', 'avoida', 'avoidb', 'avoidc', 'avoidd', 'anxietye', 'anxietya', 'anxietyb', 'anxietyc', 'anxietyd'])
# preds_df.to_csv("data/predictions.csv", index=False)

# # Evaluate your model
# mse = mean_squared_error(y_test, preds)
# mae = mean_absolute_error(y_test, preds)

# print(f"Chat transcript embeddings performance: mse={mse:.2f}, mae={mae:.2f}")
