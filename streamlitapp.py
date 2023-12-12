import base64

import tensorflow as tf
import streamlit as st
import imageio
import os
from utils import load_data,num_to_char
from modelutil import load_model
import numpy as np

#setting the layout of streamlit as wide
st.set_page_config(layout='wide')

#setting up sidebar
with st.sidebar:
    st.title("Python Project : Team Bug Catchers")
    st.info('Lip Reading and Transcribing')

st.title('Lip Reading Using DeepLearning')
#generating alist of options or videos
options= os.listdir(os.path.join('data','s1'))
selected_video = st.selectbox('Choose a video',options)

#generate two columns
col1,col2 =st.columns(2)

if options:
    #renderiing the input video
    with col1:
        st.info('Input_video')
        filepath = os.path.join('data', 's1', selected_video)
        os.system(f'ffmpeg -i {filepath} -vcodec libx264 test_video.mp4 -y')

        #rendering the video inside the web
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)






    with col2:
        st.info("Transcript of the Input Video:")
        video, annotations = load_data(tf.convert_to_tensor(filepath))
        conv1 = tf.strings.reduce_join(num_to_char(annotations)).numpy().decode('utf-8')
        st.text(conv1)

        st.info("This is the post-processed gif of your input video. This is what our model (kinda) sees when making predictions")
        imageio.mimsave('animation.gif', video, duration=100)
        st.image('animation.gif',width=350)

        st.info("Model   Predictions: ")
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        #convert to text
        conv= tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(conv)
