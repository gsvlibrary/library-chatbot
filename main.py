import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

class Message(BaseModel):
    message: str

# NLTK setup
nltk_data_dir = './nltk_data'
os.makedirs(nltk_data_dir, exist_ok=True)
os.environ['NLTK_DATA'] = nltk_data_dir

nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)

lemmatizer = WordNetLemmatizer()

# Load model and data
try:
    model = tf.keras.models.load_model('chatbot_model.h5')
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

try:
    with open('intents.json', encoding="utf8") as file:
        intents = json.load(file)
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
except Exception as e:
    raise RuntimeError(f"Error loading data files: {e}")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm sorry, I don't have an answer for that."

def chatbot_response(msg):
    ints = predict_class(msg, model)
    try:
        return getResponse(ints, intents)
    except Exception as e:
        return f"Error in response generation: {str(e)}"

# FastAPI App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"hello": "python"}

def decrypt(msg: str) -> str:
    return msg.replace("+", " ")

@app.post("/chat")
async def chat_with_bot(msg: Message):
    try:
        user_message = decrypt(msg.message)
        greetings = ["hello", "hi", "hey"]
        if user_message.lower() in greetings:
            return {
                "success": True,
                "message": "Greeting detected",
                "data": random.choice(["Hello!", "Hi there!", "Hey!"])
            }

        response = chatbot_response(user_message)
        return {
            "success": True,
            "message": "Response generated successfully",
            "data": response
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"An error occurred: {str(e)}",
            "data": None
        }

if __name__ == "__main__":
    uvicorn.run(app, host="172.22.124.100", port=8000, debug=True, workers=1)
