import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
submit_df = pd.read_csv("submit.csv")
predictions_df = pd.read_csv("predictions.csv")

# for Naive Bayes
train_df['combined_text_nb'] = train_df['title'].str.lower() + ' ' + train_df['text'].str.lower()
test_df['combined_text_nb'] = test_df['title'].str.lower() + ' ' + test_df['text'].str.lower()

# for KNN
train_df['title_knn'] = train_df['title'].str.lower()
test_df['title_knn'] = test_df['title'].str.lower()

# Handling NaN values
train_df.fillna('', inplace=True)
test_df.fillna('', inplace=True)

# Splitting the training data
train_data, valid_data = train_test_split(train_df, test_size=0.2, random_state=42)
y_train = train_data['label']
y_valid = valid_data['label']

# TF-IDF Vectorization - Naive Bayes
vectorizer_nb = TfidfVectorizer(max_features=5000, stop_words=["english", "french", "spanish", "german", "italian"])
X_train_nb = vectorizer_nb.fit_transform(train_data['combined_text_nb'])
X_valid_nb = vectorizer_nb.transform(valid_data['combined_text_nb'])
X_test_nb = vectorizer_nb.transform(test_df['combined_text_nb'])

# TF-IDF Vectorization - KNN
vectorizer_knn = TfidfVectorizer(max_features=30, stop_words=["english", "french", "spanish", "german", "italian"])
X_train_knn = vectorizer_knn.fit_transform(train_data['title_knn'])
X_valid_knn = vectorizer_knn.transform(valid_data['title_knn'])
X_test_knn = vectorizer_knn.transform(test_df['title_knn'])

# Instantiating the models
model_nb = MultinomialNB()
model_knn = KNeighborsClassifier(n_neighbors=3000)

# Training the models
model_nb.fit(X_train_nb, y_train)
model_knn.fit(X_train_knn, y_train)

# Evaluating Naive Bayes on the validation set
valid_preds_nb = model_nb.predict(X_valid_nb)
valid_accuracy_nb = accuracy_score(y_valid, valid_preds_nb)
print(f"Naive Bayes Validation Accuracy: {valid_accuracy_nb*100:.4f}%")

# Evaluating KNN on the validation set
valid_preds_knn = model_knn.predict(X_valid_knn)
valid_accuracy_knn = accuracy_score(y_valid, valid_preds_knn)
print(f"KNN Validation Accuracy: {valid_accuracy_knn*100:.4f}%")

# Generating predictions on the test set
knn_preds = model_knn.predict(X_test_knn)
nb_preds = model_nb.predict(X_test_nb)

knn_probs = model_knn.predict_proba(X_test_knn)  # KNN with probability
knn_prob_threshold = 0.6

# Getting probabilities and neighbors for KNN
final_predictions = []
for i in range(len(knn_preds)):
    max_prob = np.max(knn_probs[i])  # The proportion of neighbors that agree with the prediction
    if max_prob < knn_prob_threshold:
        final_predictions.append(nb_preds[i])  # Naive Bayes prediction
    else:
        final_predictions.append(knn_preds[i])  # KNN prediction

predictions_df['label'] = final_predictions

accuracy = accuracy_score(submit_df['label'], predictions_df['label'])
print(f"Test Accuracy: {accuracy*100:.4f}%")

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(submit_df['label'], predictions_df['label'])
conf_matrix_df = pd.DataFrame(conf_matrix, index=['True Negative', 'True Positive'], columns=['Predicted Negative', 'Predicted Positive'])
print(conf_matrix_df)
