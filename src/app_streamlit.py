from io import BytesIO
import requests
from fastai.vision.all import *
import pathlib
import platform
import streamlit as st
import urllib.request as url
st.set_page_config(layout='centered')
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath


export_file_name = 'export.pkl'
classes = ['Normal', 'Covid', 'Viral Pneumonia']
path = Path(__file__).parent

def show_results(img):
    label, _, outputs = learn.predict(img)
    pred_probs = outputs.numpy()*100
    df = pd.DataFrame({'Label':classes,'Confidence':pred_probs}).set_index('Label')
    
    col1, col2 = st.beta_columns(2)
            
    with col1:
        st.subheader('Uploaded Image')
        st.image(img)
            
    with col2:   
        st.subheader('Analysis Report')
        st.table(df)
        st.info(f'Predicted Label : {label}')
    
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

st.title('Chest X-Ray Scan Classifier')
st.markdown('<hr>',unsafe_allow_html=True)
st.write('Find out if the lungs are healthy or not. Upload Chest X-Ray Scanned Image or Specify URL.')


learn = load_learner(path/'saved'/export_file_name)
st.sidebar.markdown("<h1 style='text-align: center;'>Input Section</h1><br>", unsafe_allow_html=True)

with st.sidebar.form('Form1'):
    with st.beta_expander("Upload Scanned Images"):
        uploaded_file = st.file_uploader("", type=["jpg",'png','jpeg'])               
    btn1 = st.form_submit_button('Analyze')    
    

if btn1 and uploaded_file is not None:
    st.markdown('<hr>',unsafe_allow_html=True)
    with st.spinner(text='Analyzing'):        
        img = PILImage.create(uploaded_file)
        show_results(img)

st.sidebar.markdown("<h2 style='text-align: center;'>OR</h2>", unsafe_allow_html=True)

        
with st.sidebar.form('Form2'):
    with st.beta_expander("Specify URL of Scanned Images"):
        url = st.text_input('')
    btn2 = st.form_submit_button('Analyze')
        
if btn2 and url is not '':
    st.markdown('<hr>',unsafe_allow_html=True)
    with st.spinner(text='Analyzing'):
        content = requests.get(url).content
        img = BytesIO(content)
        img = PILImage.create(img)
        show_results(img)

with st.beta_expander("Dataset Link"):
    st.markdown('https://www.kaggle.com/tawsifurrahman/covid19-radiography-database')

       
