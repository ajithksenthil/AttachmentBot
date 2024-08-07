# Getting Started with Chatbot for Psychological Assessment Toolbox

Welcome to the Chatbot for Psychological Assessment Toolbox! This guide will help you set up and start using the toolbox for your psychological research. Below you'll find an overview of each script in the toolbox and detailed instructions on how to set up your environment and use the scripts.

## Overview of Scripts

### EmbeddingExtraction.py
**Purpose:** To extract numerical embeddings from the chat transcripts using OpenAI's embedding API.  
**Usage:** Run this script after `ObtainChatData.py`. It requires an API key from OpenAI, so ensure you have set this up as per the instructions in the script.

### FeatureExtraction.py
**Purpose:** To extract additional features from chat transcripts that might be relevant for your analysis.  
**Usage:** Run this script on the output of `EmbeddingExtraction.py` to enrich your data with more features.

### ChatAssessmentAnalysis.py
**Purpose:** To analyze the chat data using machine learning models, including training, validation, and testing.  
**Usage:** Use this script to train a model on your data and evaluate its performance. Make sure to adjust the script to match your dataset's features and labels.

### predict_survey_results.py
**Purpose:** To predict the survey results from a new set of chat data using the machine learning models, trained from before in ChatAssessementAnalysis.py.  
**Usage:** Use this script to predict your survey data from a new conversation.

### DataVisualization.py
**Purpose:** To create visualizations for your chat data and machine learning model results.  
**Usage:** After running `ChatAssessmentAnalysis.py`, use this script to visualize the model's performance and any other insights from your data.

## Setup Instructions

NOTE: <...> means replace with the actual thing you want to use, do not keep the "< >" just replace with what it the text in the middle represents but for you specifically
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
   - In the "Add-ons" section, search for "Heroku Postgres" and select the plan that fits your budget, Essential 0 is the cheapest.
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

### Step 6: Install Dependencies to run scripts locally

1. **Navigate to the toolbox directory in your local machine:**
   ```sh
   cd <repository-directory (replace with actual name)>
   ```
2. **Install required dependencies:**
   ```sh
   pip install -r requirements.txt
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
3. **Proceed with the analysis scripts as mentioned below.**


## Running the Scripts after you save your data from the chatbot

Each script is designed to be run independently, based on the stage of your analysis:


1. **Start by using `EmbeddingExtraction.py` to add embeddings to your data:**
   ```sh
   python scripts/EmbeddingExtraction.py
   ```
3. ** (OPTIONAL: You must implement proper data formatting) Run `FeatureExtraction.py` to include additional features:**
   ```sh
   python scripts/FeatureExtraction.py
   ```
4. **Analyze your data with `ChatAssessmentAnalysis.py`:**
   ```sh
   python scripts/ChatAssessmentAnalysis.py
   ```
5. **Specify what your new chat transcript is in the script where it says to in the comments and analyze your data with `predict_survey_results.py`:**
   ```sh
   python scripts/predict_survey_results.py
   ```
6. **Finally if you used the script predict_survey_results.py and saved it in a data frame, visualize your results using `DataVisualization.py` and specifying the correct corresponding file path:**
   ```sh
   python scripts/DataVisualization.py
   ```


## Notes for Users

- Ensure that you update file paths and column names in the scripts to match your dataset.
- The toolbox is designed to be flexible. Feel free to modify the scripts according to your research needs.
- By modifying the initial system messages in chatbot.py in ChatBot, you can modify the behavior, assign new roles and give whatever behavior you need for your research purpose by changing initial system messages.
- By modifying the survey questions and types in chatbot.py in ChatBot, you can add whatever survey methodology you like for your research needs. 
- Regularly check for updates or enhancements to the toolbox.

Happy researching!

This `README.md` file provides comprehensive instructions for setting up the Hugging Face space, configuring the Heroku database, setting up the OpenAI account, adding secrets to Hugging Face, installing dependencies, and running the scripts in the toolbox.
