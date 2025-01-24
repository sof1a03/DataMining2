import os
import pandas as pd

# Define the base folder paths
base_folder = 'negative_polarity'
deceptive_folder = os.path.join(base_folder, 'deceptive_from_MTurk')
truthful_folder = os.path.join(base_folder, 'truthful_from_Web')

# Function to gather data from the specified folders
def gather_reviews(folder_path, folds, label):
    reviews = []
    for fold in folds:
        fold_path = os.path.join(folder_path, fold)
        for filename in os.listdir(fold_path):
            file_path = os.path.join(fold_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                review_text = file.read().strip()
                reviews.append((review_text, label))
    return reviews

# Specify which folders to use for training and testing
train_folds = ['fold1', 'fold2', 'fold3', 'fold4']
test_folds = ['fold5']

# Gather reviews for training set
train_reviews = gather_reviews(deceptive_folder, train_folds, 0)  # Label 0 for deceptive
train_reviews += gather_reviews(truthful_folder, train_folds, 1)   # Label 1 for truthful

# Gather reviews for testing set
test_reviews = gather_reviews(deceptive_folder, test_folds, 0)     # Label 0 for deceptive
test_reviews += gather_reviews(truthful_folder, test_folds, 1)     # Label 1 for truthful

# Convert to DataFrames and save as CSV
train_df = pd.DataFrame(train_reviews, columns=['Review', 'Label'])
test_df = pd.DataFrame(test_reviews, columns=['Review', 'Label'])

train_df.to_csv('train_reviews.csv', index=False, encoding='utf-8')
test_df.to_csv('test_reviews.csv', index=False, encoding='utf-8')

print("CSV files 'train_reviews.csv' and 'test_reviews.csv' have been created.")
