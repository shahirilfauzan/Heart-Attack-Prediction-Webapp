# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:43:02 2022

@author: Shah
"""

import pickle
import os
import streamlit as st
import numpy as np
from PIL import Image


MODEL_PATH = os.path.join(os.getcwd(),'model','model.pkl')
with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)
    
    
    

#%% APP DEVELOPMENT

st.markdown("<h1 style='text-align: center; color: grey;'>Heart Attack Prediction App</h1>", unsafe_allow_html=True)
st.write('This application is to detect whether you have high chance of having heart attack')
image = Image.open(os.path.join(os.getcwd(),'static','HA.png'))
st.image(image, use_column_width=True)


st.sidebar.header("Please fill in the details below")
age = st.sidebar.number_input('Age',0)
cp = st.sidebar.selectbox('Chest Pain Type ( 0 = asymptomatic ; 1 = typical angina; 2 = atypical angina; 3 = non-anginal pain)',(0,1,2,3))
trtbps = st.sidebar.slider("resting blood pressure (in mm Hg)",0,250,130,1)
thalachh = st.sidebar.slider("maximum heart rate achieved",0,250,153,1)
exng = st.sidebar.selectbox('exercise induced angina (0 = no;1 = yes)',(0,1))
oldpeak = st.sidebar.slider("ST depression induced by exercise relative to rest",0.0,7.0,0.8,0.1)
caa = st.sidebar.slider("number of major vessels",0,3,2,1)
thall = st.sidebar.selectbox('thalassemia (1 = fixed defect; 2 = normal; 3 = reversable defect)',(1,2,3))
    
# Every form must have a submit button.
submitted = st.sidebar.button("Submit")
if submitted:
    new_data = np.expand_dims([age,cp,trtbps,
                               thalachh,exng,
                               oldpeak,caa,thall],axis=0)
    outcome = model.predict(new_data)[0]
        
    if outcome == 0:
            st.markdown("<h1 style='text-align: center; color: gray;'>YOUR RESULT</h1>", unsafe_allow_html=True)
            st.markdown("<h1 style='text-align: center; color: green;'>Congrats you are healthy, Keep it up!!</h1>", unsafe_allow_html=True)
            image = Image.open(os.path.join(os.getcwd(),'static','healthy.jpg'))
            st.image(image, use_column_width=True)
            st.balloons()
    else:
            st.markdown("<h1 style='text-align: center; color: gray;'>YOUR RESULT</h1>", unsafe_allow_html=True)
            st.markdown("<h1 style='text-align: center; color: red;'>You have a high chances of having heart disease</h1>", unsafe_allow_html=True)
            image = Image.open(os.path.join(os.getcwd(),'static','prevent.png'))
            st.image(image, use_column_width=True)
            st.snow()
            
st.markdown("<h1 style='text-align: center; color: grey;'>Heart Attack</h1>", unsafe_allow_html=True)
st.write('WHO estimates that more than 17.5 million people died of \
         cardiovascular diseases such as heart attack or stroke in 2012.\
        Contrary to popular belief, more than 3 out of 4 of these deaths\
        occurred in low- and middle-income countries, and men and women were\
        equally affected.A heart attack occurs when the flow of blood to the\
        heart is severely reduced or blocked. The blockage is usually due to a\
        buildup of fat, cholesterol and other substances in the\
        heart (coronary) arteries. The fatty, cholesterol-containing\
        deposits are called plaques. The process of plaque buildup\
        is called atherosclerosis. Sometimes, a plaque can rupture and form\
        a clot that blocks blood flow. A lack of blood flow can damage or\
        destroy part of the heart muscle. A heart attack is also called a\
        myocardial infarction. Prompt treatment is needed for a heart attack\
        to prevent death.')
            