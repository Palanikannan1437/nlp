import pickle
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

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