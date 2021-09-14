import streamlit as st
import numpy as np
import pandas as pd
import time
from streamlit import caching
from PIL import Image
from preprocessing_images import *

st.title('Brain MRI segmentation application')

st.markdown("***")


st.subheader('Upload the MRI scan of the brain')
option = st.radio('',('Single MRI scan', 'Multiple MRI scans'))
st.write('You selected:', option)


if option == 'Single MRI scan':
    st.subheader('Upload the MRI scan of the brain')
    uploaded_file = st.file_uploader(' ',accept_multiple_files = False)

    if uploaded_file is not None:
        # Perform your Manupilations (In my Case applying Filters)
        #img = load_preprocess_image(uploaded_file)
        img = final_fun_1(uploaded_file)
        #img = load_preprocess_image(img)
        #st.write("Image Uploaded Successfully")
        st.write(img)
        img = load_preprocess_image(str(img))

        st.image(img)
        
    else:
        st.write("Make sure you image is in TIF/JPG/PNG Format.")

elif option == 'Multiple MRI scans':
    st.subheader('Upload the MRI scans of the brain')
    uploaded_file = st.file_uploader(' ',accept_multiple_files = True)
    if len(uploaded_file) != 0:
        st.write("Images Uploaded Successfully")
        # Perform your Manupilations (In my Case applying Filters)
        for i in range(len(uploaded_file)):
            img = final_fun_1(uploaded_file[i])
            st.write(img)
            img = load_preprocess_image(str(img))

            st.image(img)
            
    else:
        st.write("Make sure you image is in TIF/JPG/PNG Format.")


st.markdown("***")

#st.write(' Try again with different inputs')

result = st.button(' Try again')
if result:
	
	uploaded_file = st.empty()
	predict_button = st.empty()
	caching.clear_cache()

