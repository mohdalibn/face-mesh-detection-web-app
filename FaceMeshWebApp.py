
"""

    PROJECT: Face Mess Detection Web
    MADE BY: Mohd Ali Bin Naser
    GITHUB : github.com/mohdalibn

"""

# Importing the required libraries for the project
from cv2 import circle
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

st.sidebar.title('Face Mesh App Controls')  # Sidebar Title
# st.sidebar.subheader('parameters')  # Sidebar subheading


# This function is going to resize the User selected Image or Video File so that it fits within the assigned space in the Web App
@st.cache()
def FrameResize(Frame, FrameWidth=None, FrameHeight=None, InterpolationMtd=cv2.INTER_AREA):
    FrameDimensions = None
    height, width, _ = Frame.shape

    # We simply return the Image or Video Frame if both the Width and Height are None
    if FrameWidth is None and FrameHeight is None:
        return Frame

    # This statement executes if only the FrameWidth is None
    if FrameWidth is None:
        result = FrameWidth / float(width)
        FrameDimensions = (int(width * result), FrameHeight)

    else:
        result = FrameWidth / float(width)
        FrameDimensions = (FrameWidth, int(height * result))

    # Here, we resize the frame using the calculated values above using opencv
    ResizedFrame = cv2.resize(Frame, FrameDimensions,
                              interpolation=InterpolationMtd)

    # Return the new Resized Frame
    return ResizedFrame


# Creating a Streamlit Selectbox to give the user options to select a mode

SelectAppMode = st.sidebar.selectbox('Select an App Mode',
                                     ['Face Mesh App Details',
                                         'Image Mode', 'Video Mode']
                                     )

# Executing statements according to the User's choice
if SelectAppMode == 'Face Mesh App Details':

    st.sidebar.markdown('---')

    st.markdown(

        "This is a **Face Mesh Detection Web Application** made using a Python library called **Streamlit**. The Face Mesh is detected using a pretrained model provided by Google's **MediaPipe Python library**. The Image or Video frames are processed using **OpenCV** and **Pillow**."

    )

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

    # Loading a video from the videos folder
    st.video('videos/DemoVideo2.mp4')


# Variables created for ease of typing
MPDrawing = mp.solutions.drawing_utils
MPFaceMesh = mp.solutions.face_mesh

if SelectAppMode == 'Image Mode':
    DrawingSpec = MPDrawing.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

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


if SelectAppMode == 'Video Mode':

    st.sidebar.markdown('---')
