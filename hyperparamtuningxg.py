import pandas as pd
import numpy as np
import xgboost as xgb

# for data visualization:
import matplotlib.pyplot as plt
import seaborn as sns

# for regression and hyperparameter tuning:
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

datafile_path = "data/chat_transcripts_with_embeddings_and_scores3.csv"

df = pd.read_csv(datafile_path)
df['embedding'] = df['embedding'].apply(lambda x: [float(num) for num in x.strip('[]').split(',')])

y_columns = ['avoide', 'avoida', 'avoidb', 'avoidc', 'avoidd', 'anxietye', 'anxietya', 'anxietyb', 'anxietyc', 'anxietyd']
y = df[y_columns].values

X_train, X_test, y_train, y_test = train_test_split(list(df.embedding.values), y, test_size=0.2, random_state=42)

#  Hyperparameters and their distributions for Random Search
param_dist = {
    'estimator__colsample_bytree': [0.3, 0.5, 0.7, 1],
    'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'estimator__max_depth': [3, 4, 5, 6, 7, 8],
    'estimator__alpha': [0, 0.5, 1, 5, 10, 15, 20],
    'estimator__lambda': [0, 0.5, 1, 5, 10, 15, 20],
    'estimator__n_estimators': [10, 50, 100, 150, 200]
}


xg_reg = xgb.XGBRegressor(objective ='reg:squarederror')
multioutput_reg = MultiOutputRegressor(xg_reg)

# Random search
random_search = RandomizedSearchCV(multioutput_reg, param_distributions=param_dist, n_iter=100, cv=5, 
                                   scoring='neg_mean_squared_error', n_jobs=-1, verbose=2, random_state=42)
random_search.fit(np.array(X_train).tolist(), y_train)

# Best parameters from random search
best_params = random_search.best_params_
print(best_params)
# Train your regression model with the best parameters
multioutput_reg_best = RandomizedSearchCV(multioutput_reg, param_distributions=param_dist, n_iter=100, cv=5, 
                                   scoring='neg_mean_squared_error', n_jobs=-1, verbose=2, random_state=42).fit(np.array(X_train).tolist(), y_train)

preds = multioutput_reg_best.predict(np.array(X_test).tolist())

# Save the predictions to a CSV
preds_df = pd.DataFrame(preds, columns=['avoide', 'avoida', 'avoidb', 'avoidc', 'avoidd', 'anxietye', 'anxietya', 'anxietyb', 'anxietyc', 'anxietyd'])
preds_df.to_csv("data/predictionsXG.csv", index=False)

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)

print(f"ada-002 embedding performance on chat transcripts: mse={mse:.2f}, mae={mae:.2f}")

import json

# Extract the best parameters from the random search
best_params = random_search.best_params_

# Save to a JSON file
with open("best_params.json", "w") as f:
    json.dump(best_params, f)

# import pandas as pd
# import numpy as np

# # for data visualization:
# import matplotlib.pyplot as plt
# import seaborn as sns

# # for regression and hyperparameter tuning:
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# # Hyperparameters and their distributions for Random Search
# param_dist = {
#     'n_estimators': [10, 50, 100, 150, 200],
#     'max_depth': [None, 10, 20, 30, 40, 50],
#     'min_samples_split': [2, 5, 10, 15],
#     'min_samples_leaf': [1, 2, 4, 6],
#     'max_features': ['auto', 'sqrt', 'log2']
# }

# # Initialize the model
# rfr = RandomForestRegressor()

# # Random search
# random_search = RandomizedSearchCV(rfr, param_distributions=param_dist, n_iter=100, cv=5, 
#                                    scoring='neg_mean_squared_error', n_jobs=-1, verbose=2, random_state=42)
# random_search.fit(X_train, y_train)

# # Best parameters from random search
# best_params = random_search.best_params_

# print(best_params)
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

# print(f"Chat transcript embeddings performance with Random Search: mse={mse:.2f}, mae={mae:.2f}")


