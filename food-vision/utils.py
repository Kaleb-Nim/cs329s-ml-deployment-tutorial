# Utils for preprocessing data etc 
import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path

import tensorflow as tf
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path

from tensorflow.keras.layers.experimental.preprocessing import StringLookup
# import tensorflow_io as tfio
from tensorflow import keras
import cv2 

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

base_classes = ['chicken_curry',
 'chicken_wings',
 'fried_rice',
 'grilled_salmon',
 'hamburger',
 'ice_cream',
 'pizza',
 'ramen',
 'steak',
 'sushi']

classes_and_models = {
    "model_1": {
        "classes": base_classes,
        "model_name": "food_vision_model_1_10_class_v2" # change to be your model name
    },
    "Handwriting_model":{
        'classes': "Nil",
        "model_name": "handwitten_word_recognition" # change to be your model name
    },
    "model_2": {
        "classes": sorted(base_classes + ["donut"]),
        "model_name": "efficientnet_model_2_11_classes"
    },
    "model_3": {
        "classes": sorted(base_classes + ["donut", "not_food"]),
        "model_name": "efficientnet_model_3_12_classes"
    }
}
image_width = 128
image_height = 32

def predict_json(project, region, model, instances, version=None):
    
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to Tensors.
        version (str): version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the 
            model.
    """
    # Create the ML Engine service object 
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)

    # Setup model path
    model_path = "projects/{}/models/{}".format(project, model)
    if version is not None:
        model_path += "/versions/{}".format(version)


    # Create ML engine resource endpoint and input data
    ml_resource = googleapiclient.discovery.build(
        "ml", "v1", cache_discovery=False, client_options=client_options).projects()
    instances_list = instances.numpy().tolist() # turn input into list (ML Engine wants JSON)

    input_data_json = {"signature_name": "serving_default",
                       "instances": instances_list}

    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()


    # # ALT: Create model api
    # model_api = api_endpoint + model_path + ":predict"
    # headers = {"Authorization": "Bearer " + token}
    # response = requests.post(model_api, json=input_data_json, headers=headers)

    if "error" in response:
        raise RuntimeError(response["error"])

    return response["predictions"]

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, rescale=False):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).
  """
  # Decode it into a tensor
#   img = tf.io.decode_image(filename) # no channels=3 means model will break for some PNG's (4 channels)
  img = tf.io.decode_image(filename, channels=3) # make sure there's 3 colour channels (for PNG's)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  # Rescale the image (get all values between 0 and 1)
  if rescale:
      return img/255.
  else:
      return img

def update_logger(image, model_used, pred_class, pred_conf, correct=False, user_label=None):
    """
    Function for tracking feedback given in app, updates and reutrns 
    logger dictionary.
    """
    logger = {
        "image": image,
        "model_used": model_used,
        "pred_class": pred_class,
        "pred_conf": pred_conf,
        "correct": correct,
        "user_label": user_label
    }   
    return logger



#  Steamlit 
AUTOTUNE = tf.data.AUTOTUNE
characters = ['!',
'"',
 '#',
 '&',
 "'",
 '(',
 ')',
 '*',
 '+',
 ',',
 '-',
 '.',
 '/',
 '0',
 '1',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '9',
 ':',
 ';',
 '?',
 'A',
 'B',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'J',
 'K',
 'L',
 'M',
 'N',
 'O',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'U',
 'V',
 'W',
 'X',
 'Y',
 'Z',
 'a',
 'b',
 'c',
 'd',
 'e',
 'f',
 'g',
 'h',
 'i',
 'j',
 'k',
 'l',
 'm',
 'n',
 'o',
 'p',
 'q',
 'r',
 's',
 't',
 'u',
 'v',
 'w',
 'x',
 'y',
 'z']
# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

max_len = 21
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

image_width = 128
image_height = 32

def canvas_result_processing(canvas_result,img_size=(image_width, image_height)):
    img = canvas_result.image_data.astype(np.uint8)
    rgba = tf.unstack(img, axis=-1)
    r, g, b, a = rgba[0], rgba[1], rgba[2], rgba[3]
    img_rgb = tf.stack([r, g, b], axis=-1)

    img_grayscaled = tf.image.rgb_to_grayscale(img_rgb, name=None)
    image = distortion_free_resize(img_grayscaled, img_size)
    image = tf.cast(image, tf.float32) / 255.0

    output_image = tf.expand_dims(image, axis=0) # expand image dimensions (224, 224, 3) -> (1, 224, 224, 3) 
    # display output_image
    # print(output_image.shape)
    return output_image