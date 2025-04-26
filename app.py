from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer  # Use Treebank tokenizer instead of word_tokenize
from nltk.stem import PorterStemmer
import re

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests from different ports

# Download NLTK data (only download stopwords, as Treebank tokenizer does not require punkt)
nltk.download('stopwords')

# Load model and vectorizer
classifier = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize Treebank tokenizer
tokenizer = TreebankWordTokenizer()

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Tokenize using Treebank tokenizer
    tokens = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        email_text = data.get('email', '')
        if not email_text:
            return jsonify({'error': 'No email text provided'}), 400
        
        # Preprocess and vectorize input
        processed_text = preprocess_text(email_text)
        vectorized_text = vectorizer.transform([processed_text])
        
        # Predict
        prediction = classifier.predict(vectorized_text)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
