# GenerateSampleData.py
# Purpose: To generate sample data for testing the Chatbot for Psychological Assessment Toolbox.

import pandas as pd
import numpy as np
import random

# Function to generate mock chat transcripts
def generate_chat_transcripts(n):
    sample_chats = []
    for _ in range(n):
        length = random.randint(5, 15)  # Random number of exchanges in a chat
        chat = ' '.join([f"Message {i+1}" for i in range(length)])
        sample_chats.append(chat)
    return sample_chats

# Function to generate mock questionnaire responses
def generate_questionnaire_responses(n):
    responses = []
    for _ in range(n):
        response = [random.randint(1, 5) for _ in range(10)]  # Assuming a Likert scale from 1 to 5
        responses.append(response)
    return responses

# Generate sample data
n_samples = 100  # Number of samples to generate
chat_transcripts = generate_chat_transcripts(n_samples)
questionnaire_responses = generate_questionnaire_responses(n_samples)

# Create DataFrames
df_chat = pd.DataFrame({'chathistory': chat_transcripts})
df_questionnaire = pd.DataFrame(questionnaire_responses, columns=[f'Question{i+1}' for i in range(10)])

# Save sample data to CSV
df_chat.to_csv('sample_chat_transcripts.csv', index=False)
df_questionnaire.to_csv('questionnaire_template.csv', index=False)

print("Sample data generated and saved.")
