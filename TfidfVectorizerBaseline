"""
Sentiment Analysis Using the IMDB Dataset

This program will continue to be changed to increase accuracy and efficiency.
Before running this code, make sure to install the HuggingFace library, datasets.
Binary classification (positive and negative) machine learning project trained and tested on randomly selected data from the IMDB Dataset.
TfidfVectorizer is used to turn the words into numbers and KNearestNeighbors is used to group test data with similar known classifications.
"""

import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.feature_extraction.text import TfidfVectorizer

################
#Transformation#
################

#Step 1: Data
random_index = np.random.randint(0, 24999, size = 3000)

imdb = load_dataset("stanfordnlp/imdb")

X_test_corpus = np.array(imdb['test']['text'])
X_train_data = np.array(imdb['train']['text'])
y_train = np.array(imdb['train']['label'])
y_test = np.array(imdb['test']['label'])

random_test_corpus = X_test_corpus[random_index]
random_train_data = X_train_data[random_index]
random_y_train = y_train[random_index]
random_y_test = y_test[random_index]

#Step 2: Model (Pre-Processing)
vectorizer = TfidfVectorizer()

#Step 3: Training
vectorizer.fit(random_train_data)

#Step 4: Transforming
X_test_csr = vectorizer.transform(random_test_corpus)
X_test = X_test_csr.toarray()

X_train_csr = vectorizer.transform(random_train_data)
X_train = X_train_csr.toarray()

##################
#Machine Learning#
##################

#Step 2: Model (KNN)
model = KNN(n_neighbors = 17)

#Step 3: Training
model.fit(X_train, random_y_train)

#Step 4: Prediction/Testing
test_prediction = model.predict(X_test)
accuracy = accuracy_score(random_y_test, test_prediction)
