from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import numpy as np
import requests
from bs4 import BeautifulSoup
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model and vectorizer
model = None
vectorizer = None

def load_model():
    """Load the model and vectorizer with proper error handling."""
    global model, vectorizer
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            logger.error("Model files not found")
            return False
            
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Model and vectorizer loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def extract_text_from_url(url):
    """Extract text content from a URL with improved error handling."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
            
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        if not text:
            logger.warning(f"No text content extracted from URL: {url}")
            return None
            
        return text
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        return None

def preprocess_text(text):
    """Preprocess text with improved cleaning."""
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        return text
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        return text

def analyze_news_characteristics(text):
    """Analyze news characteristics with improved detection."""
    try:
        characteristics = {
            'has_quotes': bool(re.search(r'[""](.*?)[""]', text)),
            'has_attribution': bool(re.search(r'(said|according to|reported|confirmed|announced|stated|revealed)', text.lower())),
            'has_dates': bool(re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', text)),
            'has_locations': bool(re.search(r'\b(Istanbul|Ankara|Moscow|Kyiv|Washington|London|Paris|Berlin|New York|Tokyo|Beijing)\b', text)),
            'has_officials': bool(re.search(r'\b(President|Minister|Official|Spokesperson|Government|Ministry|Department|Agency)\b', text)),
            'has_numbers': bool(re.search(r'\b\d+\b', text)),
            'has_proper_nouns': bool(re.search(r'\b[A-Z][a-z]+\b', text)),
            'has_em_dash': bool(re.search(r'—', text)),
            'has_paragraphs': len(text.split('\n\n')) > 1,
            'has_sources': bool(re.search(r'(source|according to|cited|reference)', text.lower())),
            'has_quotes_count': len(re.findall(r'[""](.*?)[""]', text))
        }
        return characteristics
    except Exception as e:
        logger.error(f"Error analyzing news characteristics: {str(e)}")
        return {}

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with comprehensive error handling."""
    if model is None or vectorizer is None:
        if not load_model():
            return jsonify({'error': 'Model not loaded. Please try again later.'}), 500

    try:
        # Get text or URL from form data
        text = request.form.get('text', '').strip()
        url = request.form.get('url', '').strip()

        if not text and not url:
            return jsonify({'error': 'No text or URL provided'}), 400

        # If URL is provided, extract text from it
        if url:
            if not url.startswith(('http://', 'https://')):
                return jsonify({'error': 'Invalid URL format. URL must start with http:// or https://'}), 400
                
            extracted_text = extract_text_from_url(url)
            if not extracted_text:
                return jsonify({'error': 'Could not extract text from URL. Please check the URL and try again.'}), 400
            text = extracted_text

        # Preprocess the text
        processed_text = preprocess_text(text)
        if not processed_text:
            return jsonify({'error': 'Could not process the provided text'}), 400

        # Analyze news characteristics
        characteristics = analyze_news_characteristics(processed_text)
        
        # Vectorize the text
        X = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = int(model.predict(X)[0])
        confidence = float(model.predict_proba(X).max())
        
        # Adjust confidence based on news characteristics
        if prediction == 1:  # If predicted as fake
            legitimate_indicators = sum([
                characteristics['has_quotes'],
                characteristics['has_attribution'],
                characteristics['has_dates'],
                characteristics['has_locations'],
                characteristics['has_officials'],
                characteristics['has_numbers'],
                characteristics['has_proper_nouns'],
                characteristics['has_em_dash'],
                characteristics['has_paragraphs'],
                characteristics['has_sources']
            ])
            
            # If article has many legitimate news characteristics, adjust prediction
            if legitimate_indicators >= 5:
                prediction = 0  # Change to real
                confidence = max(0.5, confidence - 0.2)  # Reduce confidence
        
        # Get feature names and their importance
        feature_names = vectorizer.get_feature_names_out()
        feature_importance = X.toarray()[0]
        
        # Get top keywords
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        keywords = [str(feature_names[i]) for i in top_indices if feature_importance[i] > 0]
        
        # Generate explanation
        explanation = []
        if prediction == 0:  # Real news
            explanation.append("This article appears to be legitimate news because it:")
            characteristics_found = False
            if characteristics['has_quotes']:
                explanation.append(f"• Contains {characteristics['has_quotes_count']} direct quotes")
                characteristics_found = True
            if characteristics['has_attribution']:
                explanation.append("• Includes proper attribution")
                characteristics_found = True
            if characteristics['has_dates']:
                explanation.append("• Provides specific dates")
                characteristics_found = True
            if characteristics['has_locations']:
                explanation.append("• Mentions specific locations")
                characteristics_found = True
            if characteristics['has_officials']:
                explanation.append("• References official sources")
                characteristics_found = True
            if characteristics['has_numbers']:
                explanation.append("• Includes specific numbers and statistics")
                characteristics_found = True
            if characteristics['has_proper_nouns']:
                explanation.append("• Uses proper names and titles")
                characteristics_found = True
            if characteristics['has_em_dash']:
                explanation.append("• Uses proper journalistic formatting")
                characteristics_found = True
            if characteristics['has_paragraphs']:
                explanation.append("• Has proper article structure")
                characteristics_found = True
            if characteristics['has_sources']:
                explanation.append("• Cites sources and references")
                characteristics_found = True
            if not characteristics_found:
                explanation.append("• Shows characteristics of well-structured news content")
                explanation.append("• Contains balanced and factual reporting")
        else:  # Fake news
            explanation.append("This article might be flagged as potentially misleading because it:")
            characteristics_found = False
            if not characteristics['has_quotes']:
                explanation.append("• Lacks direct quotes")
                characteristics_found = True
            if not characteristics['has_attribution']:
                explanation.append("• Missing proper attribution")
                characteristics_found = True
            if not characteristics['has_dates']:
                explanation.append("• No specific dates mentioned")
                characteristics_found = True
            if not characteristics['has_locations']:
                explanation.append("• Vague about locations")
                characteristics_found = True
            if not characteristics['has_officials']:
                explanation.append("• No official sources cited")
                characteristics_found = True
            if not characteristics['has_sources']:
                explanation.append("• No sources or references provided")
                characteristics_found = True
            if not characteristics_found:
                explanation.append("• Shows patterns commonly associated with misleading content")
                explanation.append("• Lacks typical characteristics of reliable news reporting")
        
        # Log the prediction
        logger.info(f"Prediction made - Text length: {len(text)}, Prediction: {'Fake' if prediction == 1 else 'Real'}, Confidence: {confidence:.2f}")
        
        return jsonify({
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': confidence,
            'keywords': keywords[:5],
            'explanation': '\n'.join(explanation),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your request. Please try again.'}), 500

if __name__ == '__main__':
    if load_model():
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    else:
        logger.error("Failed to start application: Model could not be loaded")


