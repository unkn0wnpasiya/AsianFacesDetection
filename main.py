from genericpath import isfile
import streamlit as slit
import time
import os.path
import gdown
import zipfile
from PIL import Image
import subprocess
from datetime import datetime as dt
from transformers import ViTFeatureExtractor, ViTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference

@slit.cache
def loadModel(model_path):
    
    classifier = VisionClassifierInference(
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k'),
        model = ViTForImageClassification.from_pretrained(model_path),
    )
    return classifier

def theModel(path):
    print('theModel Function!!')
    slit.markdown('***')
    slit.header('Real Time Prediction')
    uploaded_files = slit.file_uploader("Upload the photo here: ", accept_multiple_files=False)
    # try:
    with slit.container():
        if uploaded_files is not None:
            image = Image.open(uploaded_files)
            slit.markdown('***')
            with slit.expander("To see the uploaded photo:"):
                slit.image(image)
            if slit.button('Predict'):
                slit.markdown('***')
                with slit.spinner('Wait for it...'):
                    start = dt.now()
                    classifier = loadModel(path)
                    label = classifier.predict(img_path=uploaded_files)
                    running_secs = (dt.now() - start).microseconds
                slit.header('Results :speak_no_evil:')
                slit.success(f'Done!, predicted as a: {label}')
                slit.write(running_secs, 'μs')
    # except:
    #     slit.warning('Something went wrong :exclamation:')
    print('theModel Function is done!\n\n')

def download_weights(
    url='https://drive.google.com/uc?id=1foV-oDiGMBn2t0c3sPr0nK6e-H82HbI5',
    out="asian_faces_model.zip",
):
    print('The weights downloading function!!')
    weights = "model/asian_faces_detection_model_v1/"
    if os.path.isfile('asian_faces_model.zip') is False:
        if os.path.isdir(weights):
            print('the weight downloading function is done!')
            return weights
        with slit.spinner('downloading the model...'):
            print('Downloading the model from the Google Drive')
            gdown.download(url, out)

    if os.path.isdir(weights) is False:
        with slit.spinner('Unzing the model...'):
            with zipfile.ZipFile('asian_faces_model.zip', 'r') as zip_ref:
                zip_ref.extractall('./')
                print('unzing is done!!!')
            print('the weight downloading function is done!\n\n')
            return weights
    else:
        print('the weight downloading function is done!\n\n')
        return weights

def firstPage():
    slit.title('Asian Faces Detection')
    slit.markdown('***')
    slit.header('About Us: :+1:')
    slit.markdown('***')
    slit.text('Ziyi Luu -> 10495781 ')
    slit.text('Mohamed Bishr -> 10507845 ')
    slit.text('Boh Yee Choong -> 10484578 ')
    slit.text('Pasindu Jayasekara -> 10521966 ')
    

def main(model_path):
    # infoTeam()
    firstPage()
    theModel(model_path)
    slit.markdown('***')


if __name__=='__main__':
    Modelpath = download_weights()
    main(Modelpath)
