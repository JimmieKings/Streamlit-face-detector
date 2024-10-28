import cv2
import streamlit as st
import numpy as np
from datetime import datetime

# Load the face cascade classifier(The model file specifically trained to detect human faces, based on the Haar cascade method.)
face_cascade = cv2.CascadeClassifier('/Users/kingoriwangui/Downloads/haarcascade_frontalface_default.xml')

# Defining function to capture frames and detect faces
def detect_faces(color, scale_factor, min_neighbors, save_images):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the webcam.")
            break
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with adjustable scaleFactor and minNeighbors
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        
        # Draw rectangles around detected faces with chosen color
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Display the frame in the Streamlit app
        st.image(frame, channels="BGR")
        
        # Save image if save option is enabled and faces are detected
        if save_images and len(faces) > 0:
            img_name = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(img_name, frame)
            st.success(f"Image saved as {img_name}")
        
        # Exit condition to break the loop (use Streamlit button)
        if st.button("Stop Detection"):
            break
    
    # Release the capture and close any open resources
    cap.release()
    cv2.destroyAllWindows()

# Defining the Streamlit app
def app():
    st.title("Improved Face Detection using Viola-Jones Algorithm")
    
    # Instructions
    st.markdown("""
    **User Instructions:**
    1. Choose your desired settings using the options below.
    2. Press "Start Detection" to begin detecting faces from your webcam.
    3. You can adjust the detection sensitivity with scaleFactor and minNeighbors.
    4. Use the color picker to set the rectangle color around detected faces.
    5. Enable "Save Images" if you'd like to save snapshots of detected faces.
    6. Press "Stop Detection" to end the session.
    """)
    
    # Choose rectangle color
    color = st.color_picker("Select the rectangle color for detected faces", "#00FF00")
    color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
    
    # Add sliders for scaleFactor and minNeighbors
    scale_factor = st.slider("Scale Factor", min_value=1.1, max_value=2.0, value=1.3, step=0.1)
    min_neighbors = st.slider("Min Neighbors", min_value=3, max_value=10, value=5)
    
    # Toggle to save images with detected faces
    save_images = st.checkbox("Save images with detected faces")
    
    # Start face detection when button is pressed
    if st.button("Start Detection"):
        detect_faces(color_rgb, scale_factor, min_neighbors, save_images)

if __name__ == "__main__":
    app()
