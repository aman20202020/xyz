# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020
@author: Krish Naik
"""
#!pip install flask-ngrok
#from flask_ngrok import run_with_ngrok
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)#, template_folder='/content/Untitled Folder')

# Model saved with Keras model.save()
MODEL_PATH ='malariaN.h5'

# Load your trained model
model = load_model(MODEL_PATH)

import cv2
labels="/content/labels.txt"



    
    

#run_with_ngrok(app)
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index1.html')


@app.route("/prediction", methods=["POST"])
def prediction():

	img = request.files['img']

	img.save("img.jpg")

	image = cv2.imread("img.jpg")

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = cv2.resize(image, (64,64))

	image = np.reshape(image, (1,64,64,3))

	pred = model.predict(image)

	pred = np.argmax(pred,axis=1)

	pred = s(pred)

  

	return render_template("prediction.html", data=pred)

def s(pred):
  if pred==0:
    pred="Parasite"
  elif pred==1:
    pred="not having malaria"
  else:
    pred="none"

  return pred      

if __name__ == '__main__':
    app.run(debug=True)
