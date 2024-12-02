import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import pandas as pd
import numpy as np
from textblob import TextBlob

def analyze_text(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Severe"
    elif analysis.sentiment.polarity < 0:
        return "Mild"
    else:
        return "Moderate"