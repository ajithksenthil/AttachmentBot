# get_data.py
import psycopg2
import pandas as pd
from urllib.parse import urlparse

# Database URL
DATABASE_URL = 'postgres://u7bd1qqgkp3dat:p7e688a25309e4882b827626aa723c44bc29404e905661a9e22f18d54c752679e@c5hilnj7pn10vb.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/dcbt0aqo1e804s'

# Parse the database URL
result = urlparse(DATABASE_URL)
db_username = result.username
db_password = result.password
database = result.path[1:]
hostname = result.hostname
port = result.port

# Connect to the database
try:
    conn = psycopg2.connect(
        dbname=database,
        user=db_username,
        password=db_password,
        host=hostname,
        port=port
    )
    print("Connected to the database successfully")
except Exception as e:
    print(f"Error connecting to the database: {e}")
    exit()

cur = conn.cursor()

# Fetch all chat transcripts
cur.execute('SELECT * FROM chat_transcripts')
transcripts = cur.fetchall()

# Convert to DataFrame
columns = [desc[0] for desc in cur.description]
df = pd.DataFrame(transcripts, columns=columns)

# Save to CSV
output_path = "data/chatbotdata.csv"
df.to_csv(output_path, index=False)
print(f"Chat transcripts saved to {output_path}")

# Close the cursor and connection
cur.close()
conn.close()