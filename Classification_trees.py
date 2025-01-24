from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import string
import re
import numpy as np
from scipy.stats import lognorm
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import spacy

# Load spaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

# Load datasets
traindf = pd.read_csv("train_reviews.csv")
testdf = pd.read_csv("test_reviews.csv")
trainsentences = traindf["Review"].values
trainlabels = traindf["Label"].values
testsentences = testdf["Review"].values
testlabels = testdf["Label"].values

# Load stop words
with open("english", "r") as file:
    stop_words = set(file.read().splitlines())
with open("hotel_names.csv", "r") as file:
    hotel_names = set(file.read().splitlines())

# Preprocessing function with lemmatization
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if word not in hotel_names]
    
    # Lemmatize text using spaCy
    doc = nlp(" ".join(words))
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_punct and token.lemma_ not in stop_words and token.lemma_ not in hotel_names])
    
    return lemmatized_text

train_sentences_preprocessed = [preprocess_text(sentence) for sentence in trainsentences]
test_sentences_preprocessed = [preprocess_text(sentence) for sentence in testsentences]

# # Aligning lengths of truthful reviews with deceptive reviews
# deceptive_reviews = traindf[traindf["Label"] == 0]["Review"].str.len()
# truthful_reviews = traindf[traindf["Label"] == 1]

# shape, loc, scale = lognorm.fit(deceptive_reviews, floc=0)
# target_lengths = lognorm.rvs(shape, loc=loc, scale=scale, size=len(deceptive_reviews))

# truthful_review_lengths = truthful_reviews["Review"].str.len()
# tolerance = 10
# sampled_truthful = truthful_reviews[np.isclose(truthful_review_lengths, target_lengths[:, None], atol=tolerance).any(axis=0)]

# Combine sampled truthful and deceptive reviews
train_df_preprocessed = pd.DataFrame({
    'Review': train_sentences_preprocessed,
    'Label': trainlabels
})
test_df_preprocessed = pd.DataFrame({
    'Review': test_sentences_preprocessed,
    'Label': testlabels
})

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df_preprocessed["Label"].values)
y_test = le.transform(test_df_preprocessed["Label"].values)
print("UNIGRAMS\n")
# Vectorize using unigrams
vectorizer = CountVectorizer(ngram_range=(1,1))
X_train = vectorizer.fit_transform(train_df_preprocessed["Review"].values)
X_test = vectorizer.transform(test_df_preprocessed["Review"].values)

# parameters
param_grid = {
    'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
    'max_depth': [2, 4, 6, 8, 10],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4, 6],  # Minimum number of samples required to be at a leaf node
    'ccp_alpha': np.linspace(0, 0.1, 11) 
}


decision_tree = DecisionTreeClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='accuracy',n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Evaluate on test set
test_score = grid_search.score(X_test, y_test)
print("Test Set Score:", test_score)
# fit the model
decision_tree = DecisionTreeClassifier(ccp_alpha= 0.01, criterion= 'gini', max_depth= 8, min_samples_leaf= 6, min_samples_split= 2, random_state=42)
decision_tree.fit(X_train, y_train)
# predictions
y_pred = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nBIGRAMS\n")

# Vectorize using unigrams
vectorizer = CountVectorizer(ngram_range=(1,2))
X_train = vectorizer.fit_transform(train_df_preprocessed["Review"].values)
X_test = vectorizer.transform(test_df_preprocessed["Review"].values)

# parameters
param_grid = {
    'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
    'max_depth': [2, 4, 6, 8, 10],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4, 6],  # Minimum number of samples required to be at a leaf node
    'ccp_alpha': np.linspace(0, 0.1, 11) 
}


decision_tree = DecisionTreeClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='accuracy',n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Evaluate on test set
test_score = grid_search.score(X_test, y_test)
print("Test Set Score:", test_score)
# fit the model
decision_tree = DecisionTreeClassifier(ccp_alpha= 0.01, criterion= 'gini', max_depth= 8, min_samples_leaf= 6, min_samples_split= 2, random_state=42)
decision_tree.fit(X_train, y_train)
# predictions
y_pred = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")