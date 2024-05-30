# ObtainChatData.py
import pandas as pd

# Load & inspect dataset
input_datapath = "../data/chatbotdata.csv"
output_datapath = "../data/processed_chat_data.csv"
df = pd.read_csv(input_datapath)
print("Initial Data Loaded. Here's a preview:")
print(df.head(2))

# Data Preprocessing
df = df.dropna()
print(f"Data after cleaning: {len(df)} entries")

# Combine all transcripts from the same user into a single transcript
df_combined = df.groupby('user_id')['transcript'].apply(' '.join).reset_index()

# Save the processed data
df_combined.to_csv(output_datapath, index=False)
print(f"Processed data saved to {output_datapath}")

# Note to Users:
# - Ensure that your data file path is correctly specified in 'input_datapath'.
# - The script currently includes a basic data cleaning step (dropping missing values). Depending on your data, you may need to add more preprocessing steps.
# - The processed data is saved in CSV format. Make sure to use this processed file in subsequent analysis scripts.