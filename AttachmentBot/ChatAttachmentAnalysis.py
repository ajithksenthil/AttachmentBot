import pandas as pd
import numpy as np

# for data visualization:
import matplotlib.pyplot as plt
import seaborn as sns

# for regression:
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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

# Convert y_test to a DataFrame
y_test_df = pd.DataFrame(y_test, columns=['avoide', 'avoida', 'avoidb', 'avoidc', 'avoidd', 'anxietye', 'anxietya', 'anxietyb', 'anxietyc', 'anxietyd'])

# Save the true scores to a CSV file
y_test_df.to_csv("data/true_scores.csv", index=False)

# Train your regression model
# rfr = RandomForestRegressor(n_estimators=100)
rfr = RandomForestRegressor(max_depth=None, n_estimators=100, max_features='sqrt', min_samples_leaf=1, min_samples_split=2)
rfr.fit(X_train, y_train)

# Make predictions on the test data
preds = rfr.predict(X_test)

# Save the predictions to a CSV
preds_df = pd.DataFrame(preds, columns=['avoide', 'avoida', 'avoidb', 'avoidc', 'avoidd', 'anxietye', 'anxietya', 'anxietyb', 'anxietyc', 'anxietyd'])
preds_df.to_csv("data/predictions.csv", index=False)

# Evaluate your model
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)

print(f"Chat transcript embeddings performance: mse={mse:.2f}, mae={mae:.2f}")


# # with grid search to find better hyperparameters
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


# Mean Squared Error (MSE) is a measure of how close a fitted line is to data points.
# In the context of this task, a lower MSE means that our model's predicted attachment scores are closer to the true scores.
# An MSE of 1.32 suggests that the average squared difference between the predicted and actual scores is 1.32. 
# Since our scores are normalized between 0 and 1, this error could be considered relatively high, 
# meaning the model's predictions are somewhat off from the true values.

# Mean Absolute Error (MAE) is another measure of error in our predictions. 
# It's the average absolute difference between the predicted and actual scores.
# An MAE of 0.96 suggests that, on average, our predicted attachment scores are off by 0.96 from the true scores.
# Considering that our scores are normalized between 0 and 1, this error is also quite high, indicating that 
# the model's predictions are not very accurate. 

# Both MSE and MAE are loss functions that we want to minimize. Lower values for both indicate better model performance. 
# In general, the lower these values, the better the model's predictions are.


# Guide for adding additional features to improve performance:
# Additional Features Extraction
# To add new features to the data, you will need to create new columns in the DataFrame
# Each new feature will be a new column, which can be created by applying a function to the text data

# For example, if you were adding a feature for the length of the chat, you would do something like this:
# df['text_length'] = df['ChatTranscript'].apply(len)

# If you are using an external library to compute a feature (like NLTK for tokenization or sentiment analysis), you would need to import that library and use its functions.
# For example, to compute a sentiment score with TextBlob, you might do something like this:
# from textblob import TextBlob
# df['sentiment'] = df['ChatTranscript'].apply(lambda text: TextBlob(text).sentiment.polarity)

# Make sure to handle any potential errors or exceptions in your function.
# For example, if a chat is empty, trying to compute its length or sentiment might cause an error.

# After you've added your new features, you can include them in your model by adding them to your features array when you split the data into training and testing sets.
# For example, if 'text_length' and 'sentiment' are new features, you might do this:
# X = df[['embedding', 'text_length', 'sentiment']].values

# Always be sure to check your data after adding new features to make sure everything looks correct.


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
# -----------------------------------------------------------------
# this is the scatter plot code
# Loop over each column to generate individual scatter plots
# for idx, col in enumerate(column_names):
#     # Scatter plot
#     axes[idx//5, idx%5].scatter(y_test_df[col], preds_df[col], alpha=0.5)
    
#     # Plotting the diagonal line (indicating perfect prediction)
#     axes[idx//5, idx%5].plot([y_test_df[col].min(), y_test_df[col].max()], 
#                               [y_test_df[col].min(), y_test_df[col].max()], 
#                               'k--', lw=2)
    
#     # Setting titles and labels
#     axes[idx//5, idx%5].set_title(f"{col} - Actual vs. Predicted", fontsize=14)
#     axes[idx//5, idx%5].set_xlabel("Actual")
#     axes[idx//5, idx%5].set_ylabel("Predicted")

# # Adjust the layout and show the plots
# plt.tight_layout()
# plt.show()

# column_names = ['avoide', 'avoida', 'avoidb', 'avoidc', 'avoidd', 'anxietye', 'anxietya', 'anxietyb', 'anxietyc', 'anxietyd']

# # Create a DataFrame for the predictions
# preds_df = pd.DataFrame(preds, columns=column_names)

# # Create a DataFrame for the actual values
# y_test_df = pd.DataFrame(y_test, columns=column_names)

# # Create a 2x5 subplot grid
# fig, axes = plt.subplots(2, 5, figsize=(20, 10))

# # Loop over each column
# for idx, col in enumerate(column_names):
#     # Plot the actual values on the left column
#     sns.histplot(y_test_df[col], bins=10, ax=axes[idx//5, idx%5], color='blue', kde=True)

#     # Plot the predicted values on the right column
#     sns.histplot(preds_df[col], bins=10, ax=axes[idx//5, idx%5], color='red', kde=True)

#     # Set the title of the subplot
#     axes[idx//5, idx%5].set_title(f"{col} - actual vs predicted")

# # Add a legend
# plt.legend(labels=['actual', 'predicted'])

# # Show the plot
# plt.show()
