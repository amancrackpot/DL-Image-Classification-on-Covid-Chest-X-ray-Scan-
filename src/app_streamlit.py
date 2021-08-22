from io import BytesIO
import requests
from fastai.vision.all import *
import pathlib
import platform
import streamlit as st
STREAMLIT_THEME_BASE='light'

st.set_page_config(layout='centered')
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath


export_file_name = 'export.pkl'
classes = ['Normal', 'Covid', 'Viral Pneumonia']
path = Path(__file__).parent

if 'learn' not in st.session_state :
    st.session_state.learn = load_learner(path/'saved'/export_file_name)

def show_results(img):
    label, _, outputs = st.session_state.learn.predict(img)
    pred_probs = list(outputs.numpy()*100)
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


st.sidebar.title('Configurations')
st.sidebar.write('')
st.sidebar.write('')
menu = ['Demo','Upload', 'URL']
choice = st.sidebar.selectbox("Select Image Source", menu)
st.sidebar.write('')
st.sidebar.write('')
cont = st.sidebar.beta_container()
st.sidebar.write('')
st.sidebar.write('')
btn = st.sidebar.button('Analyze')

if choice == 'Upload':
    uploaded_file = cont.file_uploader("Upload an Image...", type=["jpg",'png','jpeg'])
    
    if btn and uploaded_file is not None:
        with st.spinner(text='Analyzing...'):
            try:
                img = PILImage.create(uploaded_file)
                show_results(img)
            except:
                st.error('Invalid File uploaded')

elif choice == 'URL':
    url = cont.text_input("Specify Image URL...")
        
    if btn and url is not '':
        with st.spinner(text='Analyzing...'):
            try:
                content = requests.get(url).content
                img = BytesIO(content)
                img = PILImage.create(img)
                show_results(img)
            except:
                st.error('URL specified is invalid')

else:
    cont.write('Runs demo on a sample X-Ray Image having COVID')
    url = 'https://drive.google.com/uc?export=download&id=1HxT1amw9pXBByGpjPr2Erh6TfzmM0bsN'
    if btn:
        with st.spinner(text='Analyzing...'):
            content = requests.get(url).content
            img = BytesIO(content)
            img = PILImage.create(img)
            show_results(img)

    
with st.beta_expander("Dataset Link"):
    st.markdown('https://www.kaggle.com/tawsifurrahman/covid19-radiography-database')
   
