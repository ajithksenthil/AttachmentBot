# Getting Started with Chatbot for Psychological Assessment Toolbox

Welcome to the Chatbot for Psychological Assessment Toolbox! This guide will help you set up and start using the toolbox for your psychological research. Below you'll find an overview of each script in the toolbox and detailed instructions on how to set up your environment and use the scripts.

## Overview of Scripts

### ObtainChatData.py
**Purpose:** To collect and preprocess chat transcripts for analysis.  
**Usage:** Run this script first to prepare your chat data. Make sure your chat transcripts are in a CSV file and update the `input_datapath` in the script to point to your data file.

### EmbeddingExtraction.py
**Purpose:** To extract numerical embeddings from the chat transcripts using OpenAI's embedding API.  
**Usage:** Run this script after `ObtainChatData.py`. It requires an API key from OpenAI, so ensure you have set this up as per the instructions in the script.

### FeatureExtraction.py
**Purpose:** To extract additional features from chat transcripts that might be relevant for your analysis.  
**Usage:** Run this script on the output of `EmbeddingExtraction.py` to enrich your data with more features.

### ChatAssessmentAnalysis.py
**Purpose:** To analyze the chat data using machine learning models, including training, validation, and testing.  
**Usage:** Use this script to train a model on your data and evaluate its performance. Make sure to adjust the script to match your dataset's features and labels.

### DataVisualization.py
**Purpose:** To create visualizations for your chat data and machine learning model results.  
**Usage:** After running `ChatAssessmentAnalysis.py`, use this script to visualize the model's performance and any other insights from your data.

## Setup Instructions

### Step 1: Clone the Repository
Clone the toolbox repository to your local machine:
```sh
git clone <repository-url>
cd <repository-directory>


Step 2: Setting up Hugging Face Space

	1.	Create a new Hugging Face space: Go to Hugging Face Spaces and create a new space.
	2.	Upload the required files: Upload chatbot.py, requirements.txt, and README.md to the space.
	3.	Set the metadata: Ensure your README.md has the following metadata at the top:


---
title: ExampleHostedChatBot
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
app_file: chatbot.py
pinned: false
license: mit
---

Step 3: Setting up the Database in Heroku

	1.	Create a Heroku account: If you don’t have one, sign up at Heroku.
	2.	Create a new app:
        heroku create examplechatbot
    3.	Add PostgreSQL addon:
        heroku addons:create heroku-postgresql:hobby-dev -a examplechatbot
    4.	Retrieve the database URL:
        heroku config:get DATABASE_URL -a examplechatbot
Step 4: Setting up OpenAI Account

	1.	Create an OpenAI account: If you don’t have one, sign up at OpenAI.
	2.	Generate an API key: Go to the API keys section and create a new key.

Step 5: Configuring Hugging Face Secrets

	1.	Add secrets to Hugging Face space:
	•	DATABASE_URL with the value obtained from Heroku.
	•	OPENAI_API_KEY with your OpenAI API key.

Step 6: Install Dependencies
	1.	Navigate to the toolbox directory:
        cd <repository-directory>
	2.	Install required dependencies:
        pip install -r requirements.txt

Running the Scripts

Each script is designed to be run independently, based on the stage of your analysis:

	1.	Start by running ObtainChatData.py to prepare your data:

        python scripts/ObtainChatData.py

    2.	Use EmbeddingExtraction.py to add embeddings to your data:

        python scripts/EmbeddingExtraction.py

	3.	Run FeatureExtraction.py to include additional features:

        python scripts/FeatureExtraction.py

	4.	Analyze your data with ChatAssessmentAnalysis.py:

        python scripts/ChatAssessmentAnalysis.py

	5.	Finally, visualize your results using DataVisualization.py:

        python scripts/DataVisualization.py

Notes for Users

	•	Ensure that you update file paths and column names in the scripts to match your dataset.
	•	The toolbox is designed to be flexible. Feel free to modify the scripts according to your research needs.
	•	Regularly check for updates or enhancements to the toolbox.

Happy researching!

This `README.md` file provides comprehensive instructions for setting up the Hugging Face space, configuring the Heroku database, setting up the OpenAI account, adding secrets to Hugging Face, installing dependencies, and running the scripts in the toolbox.
