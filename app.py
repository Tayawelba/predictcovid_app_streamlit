#Import librairies
import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import load_img,img_to_array
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

#title
st.title("Image Classification")

#load image
upload_file = st.sidebar.file_uploader("Telecharger un fichier", type=['jpg','jpeg','png'])

generate_pred = st.sidebar.button("Predict")

model = tf.keras.models.load_model("model.h5")

covid_classes = {'COVID19':0,'NORMAL':1,'PNEUMONIA':2,'TUBERCULOSIS':3}

if upload_file:
    st.image(upload_file,caption="Image Telechargee",use_column_width=True)
    test_image = image.load_img(upload_file,target_size=(64,64))
    image_array = img_to_array(test_image)
    image_array = np.expand_dims(image_array,axis=0)

    if generate_pred:
        predictions = model.predict(image_array)
        classes = np.argmax(predictions[0])
        for key,value in covid_classes.items():
            if value == classes:
                st.write("The diagnostic is :",key)