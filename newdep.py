#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[2]:


# give a title to our app 
st.title('Welcome to AR app') 
st.write ("""
This app predict **The Student's Academic** Results!
""")
st.sidebar.header('User Input Features')


# In[3]:


def user_input_features():
        G1=st.sidebar.slider('1st grade',0,20,10)
        G2=st.sidebar.slider('2nd grade',0,20,10)
        G3=st.sidebar.slider('3rd grade',0,20,10)
        Pstatus = st.sidebar.selectbox('Parent status:(A/T)',(0,1))
        romantic=st.sidebar.selectbox('Relationship status:(N/Y)',(0,1))
        Dalc=st.sidebar.slider('Workday ALcohol consumption',0,5)
        Walc=st.sidebar.slider('Weekend Alcohol consumption',0,5)
        data = {'Pstatus': Pstatus,
                'romantic':romantic,
                'Dalc': Dalc,
                'Walc': Walc,
                'G1': G1,
                'G2': G2,
                'G3': G3}
        features = pd.DataFrame(data, index=[0])
        return features
input_df = user_input_features()
st.subheader('Student input parameters')
st.write(input_df)


# In[4]:


student_raw=pd.read_csv("student_input.csv",header=0)
student_raw


# In[5]:


import os

# Check the current working directory
print("Current working directory:", os.getcwd())


# In[6]:


# Check if the file exists
file_path = 'student_input.csv'
if os.path.exists(file_path):
    print("File exists!")
else:
    print("File does not exist. Check the file path.")


# In[7]:


x=student_raw.drop(["final_grade"],axis=1)
y=student_raw.final_grade
y


# In[8]:


clf=RandomForestClassifier()
clf.fit(x,y)


# In[9]:


prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)


# In[10]:


st.subheader('Prediction')
final_grade = np.array(['good','fair','poor'])
st.write("Academic Result  ",prediction)

st.subheader('Prediction Probability')
st.write("prediction probability ",prediction_proba)


# In[11]:


if __name__ == '__user_input_features__':
    user_input_features()

