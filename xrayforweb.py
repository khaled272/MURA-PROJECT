'''
This code is to allowing the users to have an interface to test their xray 
images.  It make the predictions and then connect to (/static/predict.html) 
using flask to output the results
'''

import keras
from keras.preprocessing import image
import base64
import numpy as np
import io
from PIL import Image
from flask import request
from flask import jsonify
from flask import Flask
import tensorflow as tf
from keras.models import model_from_json

global graph
graph =tf.get_default_graph()

# Set the identification dictionary
classification_dic={0:'ELBOW', 1:'FINGER', 2:'FOREARM',3:'HAND',4:'HUMERUS',5:'SHOULDER',6:'WRIST'}
abnormality_dic={0:'negative', 1:'positive'}

# Function to load the models
def get_model():
    global ULC_model
    global elbow_model
    global finger_model
    global forearm_model
    global hand_model
    global humerus_model
    global shoulder_model
    global wrist_model
   
    #(1)
    # Upper Limb Classification model aschitecture
    print('Loadind ULC model...')
    with open('h5/model_architecture_ULCwAnshu.json', 'r') as f:
        ULC_model = model_from_json(f.read())
    # Upper Limb Classification model weights
    ULC_model.load_weights('h5/modelMN_ULC_best03w Anshu.h5')
    
    #(2)
    print('Loadind Elbow model...')
    # Elbow Abnormality detection model aschitecture
    with open('h5/model_architectureXR_ELBOW final.json', 'r') as f:
        elbow_model = model_from_json(f.read())
    # Elbow Abnormality detection model weights
    elbow_model.load_weights('h5/modelMNXR_ELBOW final.h5')
    
    #(3)
    print('Loadind Finger model...')
    # Finger Abnormality detection model aschitecture
    with open('h5/model_architectureXR_FINGER_final.json', 'r') as f:
        finger_model = model_from_json(f.read())
    # Finger Abnormality detection model weights
    finger_model.load_weights('h5/modelMNXR_FINGER_final.h5')
    
    #(4)
    print('Loadind Forearm model...')
    # Forearm Abnormality detection model aschitecture
    with open('h5/model_architectureXR_FOREARM_final.json', 'r') as f:
        forearm_model = model_from_json(f.read())
    # Forearm Abnormality detection model weights
    forearm_model.load_weights('h5/modelMNXR_FOREARM_final.h5')
    
    #(5)
    print('Loadind Hand model...')
    # Hand Abnormality detection model aschitecture
    with open('h5/model_architectureXR_HAND_final1.json', 'r') as f:
        hand_model = model_from_json(f.read())
    # Hand Abnormality detection model weights
    hand_model.load_weights('h5/modelMNXR_HAND_final1.h5')
    
    #(6)
    print('Loadind Humerus model...')
    # Humerus Abnormality detection model aschitecture
    with open('h5/model_architectureXR_HUMERUS_final.json', 'r') as f:
        humerus_model = model_from_json(f.read())
    # Humerus Abnormality detection model weights
    humerus_model.load_weights('h5/modelMNXR_HUMERUS_final.h5')
    
    #(7)
    print('Loadind Shoulder model...')
    # Shoulder Abnormality detection model aschitecture
    with open('h5/model_architectureXR_SHOULDERfinal2.json', 'r') as f:
        shoulder_model = model_from_json(f.read())
    # Shoulder Abnormality detection model weights
    shoulder_model.load_weights('h5/modelMNXR_SHOULDERfinal2.h5')
    
    #(8)
    print('Loadind Wrist model...')
    # Wrist Abnormality detection model aschitecture
    with open('h5/model_architecture_XR_WRIST_final.json', 'r') as f:
        wrist_model = model_from_json(f.read())
    # Wrist Abnormality detection model weights
    wrist_model.load_weights('h5/model_MNXR_WRIST_final.h5')
    
    print('All models are downloaded!!')

    
get_model()

# Preparing the images for predictions
def prepare_image(file):
    img=file
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

# Using Flask API to connect WEB
app = Flask(__name__)

# This decorator will uses the jsonified output of the predict() function
# when (/predict) is typed in the browser
@app.route("/predict",methods=['post','get'])

def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))  # PIL.PngImagePlugin.PngImageFile
    
    # Predict which PART of the upper limb
    preprocessed_image = prepare_image(image)
    with graph.as_default():
        predictions = ULC_model.predict(preprocessed_image)
    
    elbow = predictions[0][0]
    finger = predictions[0][1]
    forearm = predictions[0][2]
    hand = predictions[0][3]
    humerus = predictions[0][4]
    shoulder = predictions[0][5]
    wrist = predictions[0][6]
    part_pred=classification_dic[np.argmax(predictions, axis=1)[0]]
    
    # Use the part identification to pick the corresponding model for
    # abnormality detection
    if part_pred == 'ELBOW': 
        preprocessed_image = prepare_image(image)
        with graph.as_default():
            predictions = elbow_model.predict(preprocessed_image)
    if part_pred == 'FINGER': 
        preprocessed_image = prepare_image(image)
        with graph.as_default():
            predictions = finger_model.predict(preprocessed_image)
    if part_pred == 'FOREARM': 
        preprocessed_image = prepare_image(image)
        with graph.as_default():
            predictions = forearm_model.predict(preprocessed_image)
    if part_pred == 'HAND': 
        preprocessed_image = prepare_image(image)
        with graph.as_default():
            predictions = hand_model.predict(preprocessed_image)
    if part_pred == 'HUMERUS': 
        preprocessed_image = prepare_image(image)
        with graph.as_default():
            predictions = humerus_model.predict(preprocessed_image)
    if part_pred == 'SHOULDER': 
        preprocessed_image = prepare_image(image)
        with graph.as_default():
            predictions = shoulder_model.predict(preprocessed_image)
    if part_pred == 'WRIST': 
        preprocessed_image = prepare_image(image)
        with graph.as_default():
            predictions = wrist_model.predict(preprocessed_image)  
    neg = predictions[0][0]
    pos = predictions[0][1]
    pred=abnormality_dic[np.argmax(predictions, axis=1)[0]]
        
    # Post the result
    response = {
        'prediction':{
                'elbow': round(elbow.item(), 4),
                'finger': round(finger.item(), 4),
                'forearm': round(forearm.item(), 4),
                'hand': round(hand.item(), 4),
                'humerus': round(humerus.item(), 4),
                'shoulder': round(shoulder.item(), 4),
                'wrist': round(wrist.item(), 4),                
                'neg': round(neg.item(), 4),
                'pos': round(pos.item(), 4),
                'part': part_pred,
                'pred': pred
                
                
        }
    }
        
        
    return jsonify(response)
