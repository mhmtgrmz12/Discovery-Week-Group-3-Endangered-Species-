import streamlit as st
import cv2
import numpy as np
from PIL import Image
from main import predict_image

st.title("Endangered Animal Identification System")
st.write("Identify an animal by turning on your camera.")

# Kamera açma butonu
run_camera = st.checkbox("Start Camera")

if run_camera:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.write("Something went wrong.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image)

        class_name, confidence_score = predict_image(img_pil)

        cv2.putText(frame, f"{class_name} ({confidence_score*100:.2f}%)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()
