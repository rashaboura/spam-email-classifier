
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Download NLTK stopwords
nltk.download('stopwords')

# Use Treebank tokenizer (does not require punkt)
tokenizer = TreebankWordTokenizer()

# Load the dataset
df = pd.read_csv('emails.csv')  # Update path if needed

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize using Treebank tokenizer
    tokens = tokenizer.tokenize(text)
    
    # Remove stopwords and stem
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['processed_text'])
y = df['spam']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Evaluate model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.show()

# Class Distribution (to check if dataset is balanced)
plt.figure(figsize=(6, 4))
df['spam'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Class Distribution (Spam vs Not Spam)')
plt.xticks([0, 1], ['Not Spam', 'Spam'], rotation=0)
plt.ylabel('Count')
plt.savefig('class_distribution.png')
plt.show()

# ROC Curve
y_prob = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.show()

# Save model and vectorizer
joblib.dump(classifier, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Function to classify new email
def classify_email(email_text):
    classifier = joblib.load('spam_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    processed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = classifier.predict(vectorized_text)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Example usage
sample_email = "Subject: Win a free iPhone now! Click here to claim your prize!"
result = classify_email(sample_email)
print(f"Sample email classification: {result}")
