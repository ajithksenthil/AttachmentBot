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

1. **Open Terminal or Command Prompt:**  
   On Windows, you can open Command Prompt or PowerShell. On Mac and Linux, you can use Terminal.
2. **Run the following commands:**
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

### Step 2: Setting up Hugging Face Space

1. **Create a Hugging Face account and organization:**
   - Go to [Hugging Face](https://huggingface.co) and sign up for an account if you don't already have one.
   - After logging in, create a new organization by clicking on your profile picture in the top right corner and selecting "New organization."

2. **Create a new space within the organization:**
   - Go to your organization page and click on "New Space."
   - Choose a name for your space and select the "Gradio" SDK.

3. **Upload the required files:**
   - Navigate to the `Chatbot` folder in the main directory of your cloned repository.
   - Upload `chatbot.py`, `requirements.txt`, and `README.md` to the space.

4. **Set the metadata:**
   - Ensure your `README.md` has the following metadata at the top:
     ```md
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
     ```

### Step 3: Setting up the Database in Heroku

1. **Create a Heroku account:**
   - Go to [Heroku](https://www.heroku.com) and sign up for an account if you don't already have one.

2. **Create a new app:**
   - After logging in, click on "New" and select "Create new app."
   - Name your app (e.g., `examplechatbot`) and choose a region close to you.

3. **Add PostgreSQL addon:**
   - Navigate to the "Resources" tab of your new app.
   - In the "Add-ons" section, search for "Heroku Postgres" and select the "Hobby Dev - Free" plan.
   - Click on "Provision" to add the database to your app.

4. **Retrieve the database URL:**
   - Go to the "Settings" tab of your app.
   - Click on "Reveal Config Vars" and find the `DATABASE_URL` value.
   - Copy the `DATABASE_URL` as you will need it later.

### Step 4: Setting up OpenAI Account

1. **Create an OpenAI account:**
   - Go to [OpenAI](https://www.openai.com) and sign up for an account if you don't already have one.

2. **Generate an API key:**
   - After logging in, go to the API keys section.
   - Click on "Create new key" and copy the generated key.

### Step 5: Configuring Hugging Face Secrets

1. **Add secrets to Hugging Face space:**
   - Go to your Hugging Face space settings.
   - Add two secrets:
     - `DATABASE_URL` with the value obtained from Heroku.
     - `OPENAI_API_KEY` with your OpenAI API key.

### Step 6: Install Dependencies

1. **Navigate to the toolbox directory:**
   ```sh
   cd <repository-directory>
   ```
2. **Install required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Running the Scripts

Each script is designed to be run independently, based on the stage of your analysis:

1. **Start by running `ObtainChatData.py` to prepare your data:**
   ```sh
   python scripts/ObtainChatData.py
   ```
2. **Use `EmbeddingExtraction.py` to add embeddings to your data:**
   ```sh
   python scripts/EmbeddingExtraction.py
   ```
3. **Run `FeatureExtraction.py` to include additional features:**
   ```sh
   python scripts/FeatureExtraction.py
   ```
4. **Analyze your data with `ChatAssessmentAnalysis.py`:**
   ```sh
   python scripts/ChatAssessmentAnalysis.py
   ```
5. **Finally, visualize your results using `DataVisualization.py`:**
   ```sh
   python scripts/DataVisualization.py
   ```

## Additional File: get_data.py

### Purpose
`get_data.py` is used to interact with the database to store the chat data saved by the chatbot. This script should be run after specifying the database URL and interacting with the chatbot.

### Usage
1. **Ensure you have interacted with the chatbot to generate some data.**
2. **Run `get_data.py` to store the data in the `data` folder:**
   ```sh
   python get_data.py
   ```
3. **Proceed with the analysis scripts as mentioned above.**

## Notes for Users

- Ensure that you update file paths and column names in the scripts to match your dataset.
- The toolbox is designed to be flexible. Feel free to modify the scripts according to your research needs.
- Regularly check for updates or enhancements to the toolbox.

Happy researching!

This `README.md` file provides comprehensive instructions for setting up the Hugging Face space, configuring the Heroku database, setting up the OpenAI account, adding secrets to Hugging Face, installing dependencies, and running the scripts in the toolbox.