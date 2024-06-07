import pandas as pd
import openai
import tiktoken
import os
import config
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY")
)
# Set your OpenAI API key

# Embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"
max_tokens = 8000

# Function to get embeddings
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# Load preprocessed chat transcript data
input_datapath = "../data/chatbotdata.csv"
output_datapath = "../data/chat_transcripts_with_embeddings.csv"
df = pd.read_csv(input_datapath)

# Ensure your chat transcripts are within the token limit for embedding
encoding = tiktoken.get_encoding(embedding_encoding)
df["n_tokens"] = df["transcript"].apply(lambda x: len(encoding.encode(x)))
df = df[df["n_tokens"] <= max_tokens]

# Extract embeddings for each chat transcript
print("Extracting embeddings...")
df["embedding"] = df["transcript"].apply(lambda x: get_embedding(x, embedding_model))

# Save the data with embeddings
df.to_csv(output_datapath, index=False)
print(f"Data with embeddings saved to {output_datapath}")