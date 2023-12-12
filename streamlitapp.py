
import base64

import tensorflow as tf
import streamlit as st
import imageio
import os

import numpy as np

import os 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

def load_model() -> Sequential: 
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    model.load_weights(os.path.join('model', 'checkpoint'))

    return model



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
