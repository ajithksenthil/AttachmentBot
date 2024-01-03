# DataVisualization.py
# Purpose: Script to create visualizations for chat data and machine learning model results.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
# Assuming you have a CSV file with your model's predictions and actual scores
datafile_path = "data/model_predictions.csv"
df = pd.read_csv(datafile_path)

# Visualization Functions

def plot_feature_importances(model):
    """
    Plots feature importances of a trained model.
    """
    feat_importances = pd.Series(model.feature_importances_, index=df.columns[:-1])
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Feature Importances')
    plt.show()

def plot_actual_vs_predicted(y_actual, y_pred, title='Actual vs Predicted'):
    """
    Scatter plot for actual vs predicted values.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_actual, y=y_pred, alpha=0.6)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], '--r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

def plot_error_distribution(y_actual, y_pred, title='Error Distribution'):
    """
    Histogram for prediction errors.
    """
    errors = y_actual - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=20, kde=True)
    plt.xlabel('Prediction Error')
    plt.title(title)
    plt.show()

# Example Usage
# These are just examples. Replace 'your_model' with your actual trained model
# and 'y_actual', 'y_pred' with your actual data.

# plot_feature_importances(your_model)
# plot_actual_vs_predicted(df['ActualScore'], df['PredictedScore'])
# plot_error_distribution(df['ActualScore'], df['PredictedScore'])

# Note to Users:
# - Adjust the data paths, column names, and model variables as per your data and model.
# - Feel free to add more visualization functions based on your specific needs.
