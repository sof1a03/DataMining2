from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import string
import re
import numpy as np
from scipy.stats import lognorm

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

# Preprocessing function
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words).strip()

train_sentences_preprocessed = [preprocess_text(sentence) for sentence in trainsentences]
test_sentences_preprocessed = [preprocess_text(sentence) for sentence in testsentences]

# Aligning lengths of truthful reviews with deceptive reviews
deceptive_reviews = traindf[traindf["Label"] == 0]["Review"].str.len()
truthful_reviews = traindf[traindf["Label"] == 1]

shape, loc, scale = lognorm.fit(deceptive_reviews, floc=0)
target_lengths = lognorm.rvs(shape, loc=loc, scale=scale, size=len(deceptive_reviews))

truthful_review_lengths = truthful_reviews["Review"].str.len()
tolerance = 10
sampled_truthful = truthful_reviews[np.isclose(truthful_review_lengths, target_lengths[:, None], atol=tolerance).any(axis=0)]

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

# Set up a Pipeline for TfidfVectorizer, Chi-Squared feature selection, scaling, and Logistic Regression
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1,1))),  # Unigrams + bigrams with TF-IDF
    ('chi2', SelectKBest(chi2, k=2000)),  # Chi-Squared feature selection
    ('scaler', StandardScaler(with_mean=False)),  # Standardization
    ('classifier', LogisticRegression(penalty='l1', solver='liblinear'))
])

# Define parameter grid with finer increments for 'C'
param_grid = {
    'classifier__C': [0.1, 0.5, 1.0, 1.5, 2.0]  # Refine around the best-known values
}

# Run GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')
grid_search.fit(train_df_preprocessed["Review"].values, y_train)

# Evaluation on Training Data
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Final Model Evaluation on Test Data
y_pred = grid_search.predict(test_df_preprocessed["Review"].values)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# Non-zero Coefficients for Feature Importance
logistic_regression = grid_search.best_estimator_.named_steps['classifier']
vectorizer = grid_search.best_estimator_.named_steps['vectorizer']
chi2_selector = grid_search.best_estimator_.named_steps['chi2']

# Retrieve non-zero coefficients and feature names
non_zero_coefficients = (logistic_regression.coef_ != 0).sum()
total_coefficients = logistic_regression.coef_.size
print(f"Non-zero coefficients: {non_zero_coefficients}/{total_coefficients}")

# Feature importance analysis
coefficients = logistic_regression.coef_[0]
selected_feature_indices = chi2_selector.get_support(indices=True)
selected_feature_names = vectorizer.get_feature_names_out()[selected_feature_indices]
coef_df = pd.DataFrame({'feature': selected_feature_names, 'coefficient': coefficients})

# Display important features for genuine and deceptive reviews
genuine_features = coef_df[coef_df['coefficient'] > 0].sort_values(by='coefficient', ascending=False)
deceptive_features = coef_df[coef_df['coefficient'] < 0].sort_values(by='coefficient')

print("Important Features for Genuine Reviews:")
print(genuine_features.head())

print("\nImportant Features for Deceptive Reviews:")
print(deceptive_features.head())
