
"""

    PROJECT: Face Mess Detection Web
    MADE BY: Mohd Ali Bin Naser
    GITHUB : github.com/mohdalibn

"""

# Importing the required libraries for the project
import streamlit as st
import mediapipe as mp
from PIL import Image
import numpy as np
import tempfile
import time
import cv2


# Setting the Title of the Streamlit App
st.title('Face Mesh Detection Web App')


# Adding the SideBar Markdown
st.markdown(
    """

    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{width:350px}

    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width:350px
        margin-left: -350px
        }
    </style>

    """,

    unsafe_allow_html=True,

)

st.sidebar.title('FaceMesh Sidebar')
st.sidebar.subheader('parameters')
