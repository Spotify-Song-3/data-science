from flask import Flask, render_template, request, jsonify
import pickle
import requests
import json
import numpy as np

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Model
sid = SentimentIntensityAnalyzer()

# Pickle it
pickle.dump(sid, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))

# Create App
app = Flask(__name__)

# @app.route('/', methods = ['GET', 'POST'])
# def analysis():

#     # Take JSON input
#     text = request.get_json(force = True)
#     # Run JSON as text through model
#     prediction = model.polarity_scores(str(text))

#     # Re-convert results to JSON
#     return jsonify(results = prediction)

if __name__ == '__main__':
    app.run()