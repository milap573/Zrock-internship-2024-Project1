# Project-1: Spam Detection

# Overview
This project implements a machine learning model to classify SMS messages as either "spam" or "ham" (non-spam). It leverages natural language processing (NLP) techniques and the Multinomial Naive Bayes classifier for accurate classification.

# Dataset
The SMS spam dataset used in this project contains text messages labeled as "spam" or "ham". It was sourced from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset.

# Steps Involved
1. Data Preprocessing:
   i. Text cleaning: Lowercasing, punctuation removal.
   ii. Tokenization and removal of stopwords.
   iii. Stemming using Porter stemming algorithm.

2. Feature Engineering:
   i. TF-IDF vectorization: Converting text data into numerical features.

3. Model Selection and Training:
Utilized Multinomial Naive Bayes classifier for its suitability in text classification tasks.
Trained the model on a labeled dataset.

4. Model Evaluation:
Evaluated performance using metrics such as precision, recall, and F1-score.
Visualized results with a confusion matrix.

5. Hyperparameter Tuning:
Used GridSearchCV for optimizing model parameters like alpha for better accuracy.

6. Deployment and Usage:
Saved the trained model and TF-IDF vectorizer for future predictions on new SMS messages.

# Files Included
1. Z_Rock_ML_Internship_Project_1.ipynb: Jupyter notebook containing the entire project code and detailed explanations.
2. spam.csv: Dataset used for training and testing the model.
3. requirements.txt: List of Python dependencies required to run the project.
4. spam_detection_model.pkl: Serialized Multinomial Naive Bayes classifier trained to classify SMS messages as 'spam' or 'ham'.
5. tfidf_vectorizer.pkl: Serialized TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer used to transform text data into numerical features for SMS spam detection.

# Usage
Clone the repository using the following commands: 
1. git clone https://github.com/milap573/Zrock-internship-2024-Project1.git
2. cd Zrock-internship-2024-Project1

# Install dependencies
pip install -r requirements.txt
