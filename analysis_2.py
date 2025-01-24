import pandas as pd
import numpy as np
import re
import string
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from nltk import pos_tag
from googletrans import Translator
from sklearn.feature_selection import SelectKBest, chi2


# Helper functions for metadata extraction
def preprocess_text(text, stop_words):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words).strip()


def fetchVerbFeatures(text):
    # Add verb analysis logic here if needed
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return [sentiment['pos'], sentiment['neg'], sentiment['neu'], sentiment['compound']]


def fetchCapitalLetterRatio(text):
    sentences = [x for x in text.replace("...", ".").split(".") if x is not None]
    firstwords = [sentence.split()[0] for sentence in sentences if re.match(r'^\s*[A-Za-z]+', sentence)]
    propernouns = [word for word, tag in pos_tag(text.split()) if tag == 'NNP']
    
    ratiopropernouns = sum(1 for w in propernouns if re.match(r'^\s*[A-Z]', w)) / len(propernouns) if propernouns else 1.0
    ratiocapitalize = sum(1 for w in firstwords if re.match(r'^\s*[A-Z]', w)) / len(firstwords)
    
    return [ratiocapitalize, ratiopropernouns]


def getTextMetaData(documents):
    ret = []
    for text in documents:
        newfeats = fetchVerbFeatures(text)
        newfeats.extend(fetchCapitalLetterRatio(text))
        ret.append(newfeats)
    return np.array(ret)


# Load datasets
traindf = pd.read_csv("train_reviews.csv")
testdf = pd.read_csv("test_reviews.csv")

# Preprocessing
with open("refined_english_stopwords.txt", "r") as file:
    stop_words = set(file.read().splitlines())

traindf['Review'] = traindf['Review'].apply(lambda x: preprocess_text(x, stop_words))
testdf['Review'] = testdf['Review'].apply(lambda x: preprocess_text(x, stop_words))

# Feature extraction
vectorizer = CountVectorizer(ngram_range=(1, 2))  # Unigrams and bigrams
X_train = vectorizer.fit_transform(traindf['Review'])
X_test = vectorizer.transform(testdf['Review'])

# Add metadata features (Verb Tenses, Capitalization, Sentiment)
train_metadata = getTextMetaData(traindf['Review'])
test_metadata = getTextMetaData(testdf['Review'])

# Combine n-gram and metadata features
X_train_combined = np.hstack((train_metadata, X_train.toarray()))
X_test_combined = np.hstack((test_metadata, X_test.toarray()))

y_train = traindf['Label'].values
y_test = testdf['Label'].values

# Train and evaluate Naive Bayes with Chi-Squared feature selection
chi2_selector = SelectKBest(chi2, k=2000)
X_train_chi2 = chi2_selector.fit_transform(X_train_combined, y_train)
X_test_chi2 = chi2_selector.transform(X_test_combined)

nb_model = MultinomialNB()
nb_model.fit(X_train_chi2, y_train)
y_pred = nb_model.predict(X_test_chi2)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Naive Bayes - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

# Bootstrap statistics
bootstrap_indexes = [resample(np.arange(len(y_test)), n_samples=len(y_test)) for _ in range(10)]
def bootstrap_eval(model, X_test, y_test, bootstrap_indexes):
    metrics = []
    for idx in tqdm(bootstrap_indexes):
        y_pred = model.predict(X_test[idx])
        acc = accuracy_score(y_test[idx], y_pred)
        metrics.append(acc)
    return np.mean(metrics), np.std(metrics)

bootstrap_acc, bootstrap_std = bootstrap_eval(nb_model, X_test_chi2, y_test, bootstrap_indexes)
print(f"Bootstrap Accuracy: {bootstrap_acc} ± {bootstrap_std}")

# Grid search for hyperparameter tuning
param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_chi2, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_}")

