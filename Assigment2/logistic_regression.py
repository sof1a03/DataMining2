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
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Load spaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")
'''To maintain a comparable structure WITHIN LOGISTIC REGRESSION truthful reviews sampled to fit a log-normal distribution 
aligned with the lengths of the deceptive reviews, as truthful reviews are generally longer on average. 

Makes sense because Lasso penalizes less informative features, focusing the model on discriminative words or phrases 
instead of review length. By aligning lengths, we reduce the potential noise from length variations. 
This approach also preserves the interpretability of logistic regression, making it easier to understand which words 
contribute most to classification.'''


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

# Vectorize using unigrams
vectorizer = TfidfVectorizer(ngram_range=(1,1))
X_train = vectorizer.fit_transform(train_df_preprocessed["Review"].values)
X_test = vectorizer.transform(test_df_preprocessed["Review"].values)

# Logistic Regression with GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 5, 10]}
logistic_regression = LogisticRegression(penalty='l1', solver='liblinear')

grid_search = GridSearchCV(logistic_regression, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Results
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)
test_score = grid_search.score(X_test, y_test)
print("Test Set Score:", test_score)

# Final Model with Best Parameters
logistic_regression = LogisticRegression(penalty='l1', solver='liblinear', C=grid_search.best_params_['C'])
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Non-zero Coefficients
non_zero_coefficients = (logistic_regression.coef_ != 0).sum()
total_coefficients = logistic_regression.coef_.size
print(f"Non-zero coefficients: {non_zero_coefficients}/{total_coefficients}")

# Feature importance
coefficients = logistic_regression.coef_[0]
feature_names = vectorizer.get_feature_names_out()
coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})

# Display important features for genuine and deceptive reviews
genuine_features = coef_df[coef_df['coefficient'] > 0].sort_values(by='coefficient', ascending=False)
deceptive_features = coef_df[coef_df['coefficient'] < 0].sort_values(by='coefficient')

print("Important Features for Genuine Reviews:")
print(genuine_features.head())
print("\nImportant Features for Deceptive Reviews:")
print(deceptive_features.head())
