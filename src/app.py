from __future__ import division, print_function
import sys
import os
import glob
import re
from pathlib import Path
from io import BytesIO
import base64
import requests

from fastai.vision.all import *

# Flask utils
from flask import Flask, redirect, url_for, render_template, request

# Define a flask app
app = Flask(__name__)

path = Path(__file__).parent.parent
classes = ['Normal', 'Covid', 'Viral Pneumonia']

learn = load_learner(path/'export.pkl')

	
def model_predict(img):
    img = PILImage.create(BytesIO(img))
    outputs = learn.predict(img)[2].numpy()
    formatted_outputs = [f"{i*100:.2f}" for i in outputs]
    pred_probs = zip(classes, map(str, formatted_outputs))
	
    img_bytes  = img.to_bytes_format()
    img_data = base64.b64encode(img_bytes).decode()
    
    result = {"probs":pred_probs, "image":img_data}
    return render_template('result.html', result=result)
   

@app.route('/', methods=['GET', "POST"])
def index():
    # Main page
    return render_template('index.html')


@app.route('/upload', methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        img = request.files['file'].read()
        if img != None:
        # Make prediction
            preds = model_predict(img)
            return preds
    return 'OK'
	
@app.route("/classify-url", methods=["POST", "GET"])
def classify_url():
    if request.method == 'POST':
        url = request.form["url"]
        if url != None:
            response = requests.get(url)
            preds = model_predict(response.content)
            return preds
    return 'OK'
    

if __name__ == '__main__':
    port = os.environ.get('PORT', 8008)

    if "prepare" not in sys.argv:
        app.run(debug=False, host='0.0.0.0', port=port)
