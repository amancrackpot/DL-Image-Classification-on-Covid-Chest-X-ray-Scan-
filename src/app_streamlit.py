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
    url = 'https://storage.googleapis.com/kagglesdsdata/datasets/1357907/2258144/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/COVID/COVID-1.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210714%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210714T134141Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=6ed9b6b8bd6829ea3cfbcf875e190b73d88cd12c5375afc4c3269164ed84296f41b0b17c397b50ca0d24c5bac77d88c7440fc500729fe9077c301e28ca9f2ef16202be942fd5256fd60aea42b6e1fb1d49d46d1665348bfad4733159302bc85e540271b235fa556b63b20c2f52aeb66b898acc2dfcc8e0610125519da647dd9b2752609dea86aead5166dc32d3373a017091c40c77558170db2ca7012988ba498160e4e50700b92ec1980f1686725f0cbaea8b584d5399a316ff51985a59364b49912d83e5f7207e3b58ade3237bed492d89faf7407f7d6912d294caf086fc1517958efa12f845dd53b63a4aad38403925f78f13366a7c6f70faeac93d9e621f'
    if btn:
        with st.spinner(text='Analyzing...'):
            content = requests.get(url).content
            img = BytesIO(content)
            img = PILImage.create(img)
            show_results(img)

    
with st.beta_expander("Dataset Link"):
    st.markdown('https://www.kaggle.com/tawsifurrahman/covid19-radiography-database')
   
