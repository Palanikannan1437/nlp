import pickle
from pathlib import Path
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent
nlp = spacy.load("en_core_web_sm")

with open(f"{BASE_DIR}/model3.pkl", "rb") as f:
    model_peronality = pickle.load(f)

with open(f"{BASE_DIR}/Final_words.pkl", "rb") as f:
    Final_words = pickle.load(f)

classes = [
    "ENFJ",
    "ENFP",
    "ENTJ",
    "ENTP",
    "ESFJ",
    "ESFP",
    "ESTJ",
    "ESTP",
    "INFJ",
    "INFP",
    "INTJ",
    "INTP",
    "ISFJ",
    "ISFP",
    "ISTJ",
    "ISTP",
]

def predict_personality_pipeline(text):
    new_features = []
    for word in Final_words:
        if word in text.lower():
            new_features.append(1)
        else:
            new_features.append(0)
    pred = model_peronality.predict([new_features])
    return classes[int(pred)]

def summarize(text, n):
    # Step 1: Parse the text with spaCy
    doc = nlp(text)
    
    # Step 2: Remove stop words and punctuation
    stopwords = list(STOP_WORDS)
    words = [token.text for token in doc]
    freq = {}
    for word in words:
        if word.lower() not in stopwords and word.lower() not in punctuation:
            if word not in freq:
                freq[word] = 1
            else:
                freq[word] += 1
    
    # Step 3: Calculate word frequency and determine top words
    max_freq = max(freq.values())
    for word in freq.keys():
        freq[word] = (freq[word]/max_freq)
    sent_strength={}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in freq.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent]+=freq[word.text.lower()]
                else:
                    sent_strength[sent]=freq[word.text.lower()]
    
    # Step 4: Select top N sentences based on sentence strength
    summary_sents = nlargest(n, sent_strength, key=sent_strength.get)
    summary = " ".join([sent.text for sent in summary_sents])
    return summary