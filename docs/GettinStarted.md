# Getting Started with Chatbot for Psychological Assessment Toolbox

Welcome to the Chatbot for Psychological Assessment Toolbox! This guide will help you set up and start using the toolbox for your psychological research. Below you'll find an overview of each script in the toolbox and instructions on how to use them.

## Overview of Scripts

- ** ObtainChatData.py
Purpose: To collect and preprocess chat transcripts for analysis.
Usage: Run this script first to prepare your chat data. Make sure your chat transcripts are in a CSV file and update the input_datapath in the script to point to your data file.

- ** EmbeddingExtraction.py
Purpose: To extract numerical embeddings from the chat transcripts using OpenAI's embedding API.
Usage: Run this script after ObtainChatData.py. It requires an API key from OpenAI, so ensure you have set this up as per the instructions in the script.

- ** FeatureExtraction.py
Purpose: To extract additional features from chat transcripts that might be relevant for your analysis.
Usage: Run this script on the output of EmbeddingExtraction.py to enrich your data with more features.

- ** ChatAssessmentAnalysis.py
Purpose: To analyze the chat data using machine learning models, including training, validation, and testing.
Usage: Use this script to train a model on your data and evaluate its performance. Make sure to adjust the script to match your dataset's features and labels.

- ** DataVisualization.py
Purpose: To create visualizations for your chat data and machine learning model results.
Usage: After running ChatAssessmentAnalysis.py, use this script to visualize the model's performance and any other insights from your data.
Installation

## Before running the scripts, you need to set up your environment:

Clone the toolbox repository.

Navigate to the toolbox directory.

Install required dependencies by running pip install -r requirements.txt.
Running the Scripts

## Each script is designed to be run independently, based on the stage of your analysis:

Start by running ObtainChatData.py to prepare your data.

Use EmbeddingExtraction.py to add embeddings to your data.

Run FeatureExtraction.py to include additional features.

Analyze your data with ChatAssessmentAnalysis.py.

Finally, visualize your results using DataVisualization.py.
Notes for Users

Ensure that you update file paths and column names in the scripts to match your dataset.

The toolbox is designed to be flexible. Feel free to modify the scripts according to your research needs.

Regularly check for updates or enhancements to the toolbox.