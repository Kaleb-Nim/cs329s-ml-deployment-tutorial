### Script for CS329s ML Deployment Lec 
import os
import json
import requests
import SessionState
import streamlit as st
import tensorflow as tf
from utils import load_and_prep_image, classes_and_models, update_logger, predict_json,canvas_result_processing,decode_batch_predictions
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
import time
import base64
from pathlib import Path
from io import BytesIO
import uuid

# Setup environment credentials (you'll need to change these)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "daniels-dl-playground-4edbcb2e6e37.json" # change for your GCP key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "handwritten-recognition-356808-29840b51851f.json"

PROJECT = "handwritten-recognition-356808" # change for your GCP project
REGION = "us-central1" # change for your GCP region (where your model is hosted)

### Streamlit code (works as a straigtht-forward script) ###
st.title("Welcome to Handwritten Recognition 🍔📸")
st.header("Finally, teacher's can't say they cant read my handwriting -Kaleb")

drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
realtime_update = st.sidebar.checkbox("Update in realtime", True)


# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,

    update_streamlit=realtime_update,
    height=150,
    
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="png_export",
)

import cv2 
import numpy as np

image_width = 128
image_height = 32
img_size=(image_width, image_height)

# some datapreprocessing
if canvas_result.json_data is not None:
    image_tensor_processes = canvas_result_processing(canvas_result)
    st.write(image_tensor_processes)

@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(image, model, class_names):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    # image = load_and_prep_image(image)
    # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    # image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    # image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    # image = tf.expand_dims(image, axis=0)
    preds = predict_json(project=PROJECT,
                         region=REGION,
                         model=model,
                         instances=image)
    print(f'This is preds: {preds}')
    # pred_class = class_names[tf.argmax(preds[0])]
    # pred_conf = tf.reduce_max(preds[0])
    return image, preds

# # Pick the model version
# choose_model = st.sidebar.selectbox(
#     "Pick model you'd like to use",
#     ("Model 1 (10 food classes)", # original 10 classes
#      "Model 2 (11 food classes)", # original 10 classes + donuts
#      "Model 3 (11 food classes + non-food class)"), # 11 classes (same as above) + not_food class
#      "Handwriting model"
# )

CLASSES = classes_and_models["Handwriting_model"]["classes"]
MODEL = classes_and_models["Handwriting_model"]["model_name"] 

# # Model choice logic
# if choose_model == "Model 1 (10 food classes)":
#     CLASSES = classes_and_models["model_1"]["classes"]
#     MODEL = classes_and_models["model_1"]["model_name"]
# elif choose_model == "Model 2 (11 food classes)":
#     CLASSES = classes_and_models["model_2"]["classes"]
#     MODEL = classes_and_models["model_2"]["model_name"]
# elif choose_model == "Handwriting model":
#     CLASSES = classes_and_models["Handwriting_model"]["classes"]
#     MODEL = classes_and_models["Handwriting_model"]["model_name"]    
# else:
#     CLASSES = classes_and_models["model_3"]["classes"]
#     MODEL = classes_and_models["model_3"]["model_name"]

# Display info about model and classes
if st.checkbox("Show classes"):
    st.write(f"You chose {MODEL}, these are the classes of food it can identify:\n", CLASSES)
st.write(f"You chose {MODEL}, these are the classes of food it can identify:\n", CLASSES)

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload an image of food",
                                 type=["png", "jpeg", "jpg"])

# Setup session state to remember state of app so refresh isn't always needed
# See: https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11 
session_state = SessionState.get(pred_button=False)

# # Create logic for app flow
# if not uploaded_file:
#     st.warning("Please upload an image.")
#     st.stop()
# else:
#     session_state.uploaded_image = uploaded_file.read()
#     st.image(session_state.uploaded_image, use_column_width=True)
#     pred_button = st.button("Predict")

pred_button = st.button("Predict")
# # Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 

# And if they did...
if session_state.pred_button:
    session_state.image, session_state.pred_value= make_prediction(image_tensor_processes, model=MODEL, class_names=CLASSES)
    st.write(f"Prediction: {session_state.pred_value}")

    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox(
        "Is this correct?",
        ("Select an option", "Yes", "No"))
    if session_state.feedback == "Select an option":
        pass
    elif session_state.feedback == "Yes":
        st.write("Thank you for your feedback!")
        # Log prediction information to terminal (this could be stored in Big Query or something...)
        print(update_logger(image=session_state.image,
                            model_used=MODEL,
                            pred_class=session_state.pred_class,
                            pred_conf=session_state.pred_conf,
                            correct=True))
    elif session_state.feedback == "No":
        session_state.correct_class = st.text_input("What should the correct label be?")
        if session_state.correct_class:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something...)
            print(update_logger(image=session_state.image,
                                model_used=MODEL,
                                pred_class=session_state.pred_class,
                                pred_conf=session_state.pred_conf,
                                correct=False,
                                user_label=session_state.correct_class))

# TODO: code could be cleaned up to work with a main() function...
# if __name__ == "__main__":
#     main()