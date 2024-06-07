import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MultiLabelBinarizer
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()

# Load the trained model
model_filename = 'models/trained_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the MultiLabelBinarizer
mlb_filename = 'models/mlb.pkl'
with open(mlb_filename, 'rb') as file:
    mlb = pickle.load(file)

# Embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"
max_tokens = 8000

# Initialize OpenAI API client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to get embeddings from text
def get_embedding(text, model=embedding_model):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Function to preprocess a new transcript
def preprocess_transcript(transcript):
    encoding = tiktoken.get_encoding(embedding_encoding)
    n_tokens = len(encoding.encode(transcript))
    if n_tokens > max_tokens:
        raise ValueError("Transcript is too long for the embedding model.")
    return get_embedding(transcript, embedding_model)

# Function to predict survey results from an embedding
def predict_survey_results(embedding):
    embedding_array = np.array(embedding).reshape(1, -1)
    prediction = model.predict(embedding_array)
    survey_results = mlb.inverse_transform(prediction)
    return survey_results

# Example usage
if __name__ == "__main__":
    example_transcript = "Your example transcript here"
    embedding = preprocess_transcript(example_transcript)
    survey_results = predict_survey_results(embedding)
    print("Predicted Survey Results:", survey_results)