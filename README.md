# Naive Bayes SMS Spam Detection

## Overview
A machine learning project that classifies SMS messages as spam or ham using TF-IDF features and Multinomial Naive Bayes.

## Workflow
- Data cleaning and duplicate analysis
- Exploratory data analysis and visualizations
- Stratified train/test split
- TF-IDF feature extraction
- Baseline Multinomial Naive Bayes model
- Hyperparameter tuning with GridSearchCV
- Evaluation using precision, recall, F1-score, and confusion matrix
- Prediction function for new SMS messages

## Model
- Text vectorizer: TfidfVectorizer
- Classifier: MultinomialNB
- Hyperparameter tuning: GridSearchCV with 5-fold cross-validation
- Selected parameter: alpha = 0.01

## Test Results
- Accuracy: approximately 98.3%
- Spam precision: approximately 97.5%
- Spam recall: approximately 88.5%
- Spam F1-score: approximately 92.8%

Results are from one held-out test split and may differ on new messages.

## Installation
Install dependencies with:

`pip install -r requirements.txt`

Open `spam_detection.ipynb` in Google Colab or Jupyter Notebook to explore the complete workflow.

## Dataset
SMS Spam Collection dataset. See the original dataset source and its terms before redistribution.
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
