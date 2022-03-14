
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

    # This allows Users to import an Image file from their local Machine
    UploadImageFile = st.sidebar.file_uploader(
        'Upload an Image', type=["png", "jpg", "jpeg"])

    # This statement executes when the file upload buffer is not empty
    if UploadImageFile is not None:
        ImageFile = np.array(Image.open(UploadImageFile))

        # These next 2 lines display the original image imported by the User on the Sidebar
        st.sidebar.text('Original Image Uploaded')
        st.sidebar.image(ImageFile)

    # If the User Upload file is empty, then we use a stock image
    else:
        StockDemoImg = "images/DemoImage1.jpg"
        ImageFile = np.array(Image.open(StockDemoImg))

        st.sidebar.text('Demo Image Provided')
        st.sidebar.image(ImageFile)

    st.sidebar.markdown('---')

    # This parameter is going to allow the User to input the number of faces that they want the model to detect on an Image or Video. We are setting the default number of faces to 2(value=2) and minimum to 1 (min_value=1)
    NumFaces = st.sidebar.number_input(
        'Select the number of faces you want to detect', value=2, min_value=1)

    st.sidebar.markdown('---')

    # Creates a Slider on the Sidebar for the User to set the Detection Confidence of the Model
    DetectionConfidence = st.sidebar.slider(
        'Minimum Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)

    # st.sidebar.markdown('---')

    FaceCount = 0
    Failed = False

    # The Code below is for the Statistics Dashboard
    with MPFaceMesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=NumFaces,
            min_detection_confidence=DetectionConfidence) as FaceMesh:

        MeshProcessResults = FaceMesh.process(ImageFile)
        OutputImage = ImageFile.copy()

        # This if statement is a fail check when the model isn't able to detect faces
        if MeshProcessResults.multi_face_landmarks is not None:

            Failed = False

            # Here is the code for drawing the Face Mesh Landmarks
            for FaceLandMarks in MeshProcessResults.multi_face_landmarks:

                # We increment our FaceCount Variable
                FaceCount += 1

                MPDrawing.draw_landmarks(
                    image=OutputImage,
                    landmark_list=FaceLandMarks,
                    connections=MPFaceMesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=DrawingSpec
                )

        else:
            Failed = True

        # Displaying the Resulting Output Image on the Main Page
        st.subheader("Resulting Output Image")
        st.image(OutputImage, use_column_width=True)

        st.subheader("**Detected Faces**")
        DetectedText = st.markdown("0")

        # These if else statements display the right text accordingly
        if Failed:
            DetectedText.write(
                f"<h2 style='text-align: center; color: #8B3DFF;'>Sorry! The model is unable to detect faces. Please try using another image.</h2>", unsafe_allow_html=True)
        else:
            DetectedText.write(
                f"<h1 style='text-align: center; color: #8B3DFF;'>{FaceCount}</h1>", unsafe_allow_html=True)


if SelectAppMode == 'Video Mode':

    # This line suppress any deprecation warning that Streamlit may Output
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # This lets the User Use their Webcam as direct input into the App
    UseWebcam = st.sidebar.button('Use Webcam')
    RecordOption = st.sidebar.checkbox('Record Video')

    if RecordOption:
        st.checkbox("Recording Video....", value=True)

    stFrame = st.empty

    # This allows Users to import an Image file from their local Machine
    UploadVideoFile = st.sidebar.file_uploader(
        'Upload a Video', type=["mp4", "avi", "mov", "asf", "m4v"])

    TmpFile = tempfile.NamedTemporaryFile(delete=False)

    # This statement executes when the file upload buffer is not empty
    if not UploadVideoFile:

        # Executes if the User Clicks on the Webcam button
        if UseWebcam:
            CamVideo = cv2.VideoCapture(0)

        # If the User does use the webcam, then we use a stock video
        else:
            CamVideo = cv2.VideoCapture("videos/DemoVideo1.mp4")

    #
    else:
        TmpFile.write(UploadImageFile.read())
        CamVideo = cv2.VideoCapture(TmpFile.name)

    # Getting the Video Width, Height, and FPS
    VideoWidth = int(CamVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    VideoHeight = int(CamVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    VideoFPS = int(CamVideo.get(cv2.CAP_PROP_FPS))

    # If the Video Recording Option is selected by the User
    RecordingCodec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    # Outputing the Recorded Video into a file
    OutputVideo = cv2.Video('recording1.mp4',
                            RecordingCodec, VideoFPS, (VideoWidth, VideoHeight))

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

    # This parameter is going to allow the User to input the number of faces that they want the model to detect on an Image or Video. We are setting the default number of faces to 2(value=2) and minimum to 1 (min_value=1)
    NumFaces = st.sidebar.number_input(
        'Select the number of faces you want to detect', value=2, min_value=1)

    st.sidebar.markdown('---')

    # Creates a Slider on the Sidebar for the User to set the Detection Confidence of the Model
    DetectionConfidence = st.sidebar.slider(
        'Minimum Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)

    TrackingConfidence = st.sidebar.slider(
        'Minimum Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)

    # st.sidebar.markdown('---')

    FaceCount = 0
    Failed = False

    # The Code below is for the Statistics Dashboard
    with MPFaceMesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=NumFaces,
            min_detection_confidence=DetectionConfidence) as FaceMesh:

        MeshProcessResults = FaceMesh.process(ImageFile)
        OutputImage = ImageFile.copy()

        # This if statement is a fail check when the model isn't able to detect faces
        if MeshProcessResults.multi_face_landmarks is not None:

            Failed = False

            # Here is the code for drawing the Face Mesh Landmarks
            for FaceLandMarks in MeshProcessResults.multi_face_landmarks:

                # We increment our FaceCount Variable
                FaceCount += 1

                MPDrawing.draw_landmarks(
                    image=OutputImage,
                    landmark_list=FaceLandMarks,
                    connections=MPFaceMesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=DrawingSpec
                )

        else:
            Failed = True

        # Displaying the Resulting Output Image on the Main Page
        st.subheader("Resulting Output Video")
        st.image(OutputImage, use_column_width=True)

        st.subheader("**Detected Faces**")
        DetectedText = st.markdown("0")

        # These if else statements display the right text accordingly
        if Failed:
            DetectedText.write(
                f"<h2 style='text-align: center; color: #8B3DFF;'>Sorry! The model is unable to detect faces. Please try using another image.</h2>", unsafe_allow_html=True)
        else:
            DetectedText.write(
                f"<h1 style='text-align: center; color: #8B3DFF;'>{FaceCount}</h1>", unsafe_allow_html=True)
