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
    url = 'https://storage.googleapis.com/kagglesdsdata/datasets/576013/2000225/COVID-19_Radiography_Dataset/COVID/COVID-1000.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210703%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210703T093222Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=6c51ab28bd6d41fe088b9afa494693de6a0d6d24a7d7b8925b4590ccefdf85facfd52a948bdca2dc9c556d07dbf830c24d962f7ea6e6a95f93c511fd8296e4fb67be6e4e87048c6461c2bb654b204cb35c585673e0abbce2c99671852b4ac2bd17e2fbff598f6926c117a431f68d0a26127cc16cc8ff897283b6a425addd3feba7d0cfabb494acdd2a672f6f816264b00168b2f8d4c74a874c2ce8d7e170568987fcbdecdf223183204b943f175d0f89aa728b825690aff2b4d242f1fbe1286ae59f39e71fbd0daca1ab8aedde6834d2d4862591b64e76a0f6db86c08a03ccaac6e65cf233147b0d01279d8cdb9f73931a554e5ff2a1dbf95ddccf5726e86b59'
    if btn:
        with st.spinner(text='Analyzing...'):
            content = requests.get(url).content
            img = BytesIO(content)
            img = PILImage.create(img)
            show_results(img)

    
with st.beta_expander("Dataset Link"):
    st.markdown('https://www.kaggle.com/tawsifurrahman/covid19-radiography-database')
   
