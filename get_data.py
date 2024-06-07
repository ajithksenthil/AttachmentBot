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

# Fetch all chat transcripts along with the closest survey responses by timestamp
query = '''
SELECT 
    ct.id as transcript_id, 
    ct.user_id, 
    ct.transcript, 
    ct.timestamp as chat_timestamp, 
    sr.responses as survey_responses,
    sr.timestamp as survey_timestamp
FROM 
    chat_transcripts ct
LEFT JOIN LATERAL (
    SELECT 
        sr.responses, 
        sr.timestamp
    FROM 
        survey_responses sr
    WHERE 
        sr.user_id::VARCHAR = ct.user_id::VARCHAR
    ORDER BY 
        ABS(EXTRACT(EPOCH FROM (ct.timestamp - sr.timestamp)))
    LIMIT 1
) sr ON true;
'''
cur.execute(query)
data = cur.fetchall()

# Convert to DataFrame
columns = ['transcript_id', 'user_id', 'transcript', 'chat_timestamp', 'survey_responses', 'survey_timestamp']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
output_path = "data/chatbotdata.csv"
df.to_csv(output_path, index=False)
print(f"Chat transcripts and survey data saved to {output_path}")

# Close the cursor and connection
cur.close()
conn.close()