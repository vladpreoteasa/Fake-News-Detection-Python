import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
submit_df = pd.read_csv("submit.csv")
predictions_df = pd.read_csv("predictions.csv")

train_df['combined_text'] = train_df['title'].str.lower() + ' ' + train_df['text']
test_df['combined_text'] = test_df['title'].str.lower() + ' ' + test_df['text']

# Handling NaN values
train_df['combined_text'].fillna('', inplace=True)
test_df['combined_text'].fillna('', inplace=True)

# Splitting the training data
train_data, valid_data = train_test_split(train_df, test_size=0.2, random_state=42)
y_train = train_data['label']
y_valid = valid_data['label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words=["english", "french", "spanish", "german", "italian"])
X_train = vectorizer.fit_transform(train_data['combined_text'])
X_valid = vectorizer.transform(valid_data['combined_text'])
X_test = vectorizer.transform(test_df['combined_text'])

model = MultinomialNB()

# Cross-validation
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cross_val_scores}")
print(f"Average score: {cross_val_scores.mean():.4f}")

# Training the model
model.fit(X_train, y_train)

# Evaluate on the validation set
valid_preds = model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, valid_preds)
print(f"Validation Accuracy: {valid_accuracy*100:.4f}%")

# Generating predictions on the test set
test_preds = model.predict(X_test)
predictions_df['label'] = test_preds

accuracy = accuracy_score(submit_df['label'], predictions_df['label'])
print(f"Test Accuracy: {accuracy*100:.4f}%")

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(submit_df['label'], predictions_df['label'])
conf_matrix_df = pd.DataFrame(conf_matrix, index=['True Negative', 'True Positive'], columns=['Predicted Negative', 'Predicted Positive'])
print(conf_matrix_df)
