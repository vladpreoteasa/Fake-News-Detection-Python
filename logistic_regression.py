import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
submit_df = pd.read_csv("submit.csv")
predictions_df = pd.read_csv("predictions.csv")

train_df['combined_text'] = train_df['title'] + ' ' + train_df['text']
test_df['combined_text'] = test_df['title'] + ' ' + test_df['text']

# Handle NaN
train_df['combined_text'].fillna('', inplace=True)
test_df['combined_text'].fillna('', inplace=True)

# Splitting data for validation
train_data, valid_data = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=27)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words=["english", "french", "spanish", "german", "italian"])

X_train = vectorizer.fit_transform(train_data['combined_text'])
X_valid = vectorizer.transform(valid_data['combined_text'])
X_test = vectorizer.transform(test_df['combined_text'])

y_train = train_data['label']
y_valid = valid_data['label']

model = LogisticRegression(random_state=27, C=0.005)

# Training the model
model.fit(X_train, y_train)

# Predictions on the validation set
valid_preds = model.predict(X_valid)

valid_accuracy = accuracy_score(y_valid, valid_preds)
print(f"Validation Accuracy: {valid_accuracy*100:.4f}%")

# Retrain on the entire train.csv
X_train_full = vectorizer.transform(train_df['combined_text'])
y_train_full = train_df['label']

model.fit(X_train_full, y_train_full)

# Generate predictions on the test set
test_preds = model.predict(X_test)

# Assigning labels to the predictions.csv file
predictions_df['label'] = test_preds

correct_results_percentage = accuracy_score(submit_df['label'], predictions_df['label']) * 100
print(f"Percentage of Correct Results on Submit.csv: {correct_results_percentage:.2f}%")

classification_rep = classification_report(submit_df['label'], predictions_df['label'])
print("\nClassification Report:")
print(classification_rep)
