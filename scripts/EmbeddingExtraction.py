# EmbeddingExtraction.py
# Purpose: Script to extract embeddings from chat transcripts using OpenAI's embedding API.

# Imports
import pandas as pd
import openai
from openai.embeddings_utils import get_embedding
import config
import tiktoken

# Set your OpenAI API key
openai.api_key = config.OPENAI_API_KEY

# Embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # Encoding for text-embedding-ada-002
max_tokens = 8000  # Max token limit for text-embedding-ada-002

# Load preprocessed chat transcript data
input_datapath = "data/processed_chat_data.csv"
output_datapath = "data/chat_transcripts_with_embeddings.csv"
df = pd.read_csv(input_datapath)

# Ensure your chat transcripts are within the token limit for embedding
# Estimate for the maximum number of words would be around 1638 words (8191 tokens / 5)
encoding = tiktoken.get_encoding(embedding_encoding)
df["n_tokens"] = df["chathistory"].apply(lambda x: len(encoding.encode(x)))
df = df[df["n_tokens"] <= max_tokens]

# Extract embeddings for each chat transcript
# Note: This may take some time depending on the size of your data
print("Extracting embeddings...")
df["embedding"] = df["chathistory"].apply(lambda x: get_embedding(x, engine=embedding_model))

# Save the data with embeddings
df.to_csv(output_datapath, index=False)
print(f"Data with embeddings saved to {output_datapath}")

# Note to Users:
# - Ensure that the 'input_datapath' points to your preprocessed chat data.
# - Adjust the 'embedding_model' and 'max_tokens' parameters if using a different OpenAI model.
# - The script filters out transcripts exceeding the token limit. Adjust this if needed.
# - The resulting file will include the original data along with the extracted embeddings.
