import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to process survey responses
def process_survey_responses(survey_responses):
    responses_dict = json.loads(survey_responses.replace("'", "\""))
    flattened_responses = []
    for key, value in responses_dict.items():
        if isinstance(value, list):
            flattened_responses.extend(value)
        else:
            flattened_responses.append(value)
    return flattened_responses

# Function to calculate sentiment scores
def get_sentiment_scores(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Function to extract most common words
def get_most_common_words(text, n=10):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_words = [word for word in word_tokens if word.isalnum() and word not in stop_words]
    return Counter(filtered_words).most_common(n)

# Load and preprocess data
print("Loading and preprocessing data...")
datafile_path = "../data/chat_transcripts_with_embeddings.csv"
df = pd.read_csv(datafile_path)
df['embedding'] = df['embedding'].apply(lambda x: np.array([float(num) for num in x.strip('[]').split(',')]))
X = np.array(df['embedding'].tolist())

df['processed_survey_responses'] = df['survey_responses'].apply(process_survey_responses)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['processed_survey_responses'])

# Save the MultiLabelBinarizer
print("Saving MultiLabelBinarizer...")
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
mlb_filename = os.path.join(model_dir, 'mlb.pkl')
with open(mlb_filename, 'wb') as file:
    pickle.dump(mlb, file)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Clustering analysis
print("Performing clustering analysis...")
n_clusters = 5  # You can adjust this or use techniques like elbow method to find optimal number
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X)
df['cluster'] = cluster_labels

# Calculate silhouette score
silhouette_avg = silhouette_score(X, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.2f}")

# Visualize clusters
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('t-SNE visualization of chat transcript clusters')
plt.savefig('cluster_visualization.png')
plt.close()

# Analyze representative samples
print("Analyzing representative samples from each cluster...")
for cluster in range(n_clusters):
    cluster_samples = df[df['cluster'] == cluster]['transcript'].sample(n=3, random_state=42)
    print(f"\nCluster {cluster} representative samples:")
    for idx, sample in enumerate(cluster_samples):
        print(f"Sample {idx + 1}:")
        print(sample[:200] + "...")  # Print first 200 characters of each sample

# NLP Analysis
print("Performing NLP analysis...")
df['sentiment_polarity'], df['sentiment_subjectivity'] = zip(*df['transcript'].apply(get_sentiment_scores))

# Visualize sentiment distribution
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sentiment_polarity', y='sentiment_subjectivity', hue='cluster', palette='viridis')
plt.title('Sentiment Distribution across Clusters')
plt.savefig('sentiment_distribution.png')
plt.close()

# Extract most common words
all_text = ' '.join(df['transcript'])
most_common_words = get_most_common_words(all_text)
print("\nMost common words across all transcripts:")
print(most_common_words)

# Train and evaluate model
print("Training and evaluating model...")
xg_reg = XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.05, 
                      max_depth=4, alpha=0, reg_lambda=0.5, n_estimators=200)
multioutput_reg = MultiOutputRegressor(xg_reg)
multioutput_reg.fit(X_train, y_train)

# Validate the model
val_preds = multioutput_reg.predict(X_test)
val_mse = mean_squared_error(y_test, val_preds)
val_mae = mean_absolute_error(y_test, val_preds)
print(f"Validation MSE: {val_mse:.2f}, Validation MAE: {val_mae:.2f}")

# Save the trained model
model_filename = os.path.join(model_dir, 'trained_model.pkl')
with open(model_filename, 'wb') as file:
    pickle.dump(multioutput_reg, file)
print(f"Model trained and saved as {model_filename}")

# Feature importance analysis
feature_importance = np.mean([tree.feature_importances_ for tree in multioutput_reg.estimators_], axis=0)
feature_importance_df = pd.DataFrame({'feature': range(len(feature_importance)), 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Top 10 Most Important Features')
plt.savefig('feature_importance.png')
plt.close()

print("Analysis complete. Check the output files for visualizations and results.")
