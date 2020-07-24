# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 23:15:40 2020

@author: Admin
"""
from __future__ import division, print_function
import sys
import glob
import re
import os
import numpy as np

#keras
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#flask utils
from flask import Flask,redirect,url_for,request,render_template
from werkzeug.utils import secure_filename

#define flask app
app = Flask(__name__)

#model saved with keras model.save()
MODEL_PATH= 'model_vgg19.h5'

#load your trained model.
model= load_model(MODEL_PATH)


def model_predict(img_path, model):
    img= image.load_img(img_path, target_size=(224,224))
    
    #preprocessing the image
    x = np.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x= x/255
    x= np.expand_dims(x,axis=0)
    
    x= preprocess_input(x)
    preds= model.predict(x)
    
    preds= np.argmax(preds, axis=1)
    
    if preds==1:
        preds='The Person Is Infected With Pneumonia'
    else:
        preds='The Person Is Not Infected With Pneumonia'
        
    return preds


@app.route('/', methods=['GET'])
def index():
    #main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        #get the file from the post request
        f = request.files['file']
        
        #save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path=os.path.join(
            basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        
        #make prediction
        preds=model_predict(file_path, model)
        result= preds
        return result
    return None

if __name__=='__main__':
    app.run(debug=True)