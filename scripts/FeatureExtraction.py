# FeatureExtraction.py
# Purpose: Script to extract additional features from chat transcripts for psychological assessments.

# Imports
import pandas as pd
import numpy as np
from textblob import TextBlob

# Function to calculate sentiment polarity
def get_sentiment_polarity(text):
    return TextBlob(text).sentiment.polarity

# Function to calculate sentiment subjectivity
def get_sentiment_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Load data with embeddings
input_datapath = "data/chat_transcripts_with_embeddings.csv"
output_datapath = "data/chat_transcripts_with_features.csv"
df = pd.read_csv(input_datapath)

# Feature Extraction
# Example: Extracting sentiment polarity and subjectivity
df['sentiment_polarity'] = df['chathistory'].apply(get_sentiment_polarity)
df['sentiment_subjectivity'] = df['chathistory'].apply(get_sentiment_subjectivity)

# TODO: Add any additional feature extraction relevant to your study here.
# Example: df['feature_name'] = df['column'].apply(your_custom_function)

# Save the data with additional features
df.to_csv(output_datapath, index=False)
print(f"Data with additional features saved to {output_datapath}")

# Note to Users:
# - Ensure that 'input_datapath' points to your data file with embeddings.
# - This script uses TextBlob for sentiment analysis. Install it using 'pip install textblob' if not already installed.
# - You can add more feature extraction functions as needed for your specific research requirements.
