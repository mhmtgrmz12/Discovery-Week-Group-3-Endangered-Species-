import streamlit as st
import cv2
import numpy as np
from PIL import Image
from main import predict_image
import time

st.title("Endangered Animal Recognition System")
st.write(
    "Your camera will be continuously on. Output will only be provided when an animal is detected and accuracy is above 90%.")

# Camera start button
run_camera = st.checkbox("Start Camera")

if run_camera:
    cap = cv2.VideoCapture(0)  # Start the camera

    # Optimize camera settings to reduce delay
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Increase FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size

    stframe = st.empty()

    if not cap.isOpened():  # Show error if camera couldn't be opened
        st.error("Camera couldn't be opened! Another program might be using it.")
        run_camera = False  # End the loop

    last_process_time = time.time()
    display_frame = None

    while run_camera:
        ret, frame = cap.read()

        # If image can't be captured from camera, wait for a while and try again
        if not ret:
            st.warning("Can't get image from camera! Check the connection.")
            time.sleep(0.1)  # Reduced wait time
            continue  # Continue the loop

        # Process frames at a lower rate to reduce CPU load (every 200ms)
        current_time = time.time()
        if current_time - last_process_time > 0.2:
            # Convert image to color format
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(image)

            class_name, category, confidence_score = predict_image(img_pil)

            # Only process and display if not Environment/Human and confidence > 90%
            if not (class_name.endswith("Human") or class_name.endswith("Environment")) and confidence_score >= 0.90:

                label_text = f"{class_name} - {confidence_score * 100:.2f}%"
                # Only write to screen if a new prediction has come
                if label_text != st.session_state.get("last_label", ""):
                    st.session_state["last_label"] = label_text
                    st.write(label_text)

                # Add label to the frame
                cv2.putText(frame, label_text, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            display_frame = frame
            last_process_time = current_time

        # Always display the most recent processed frame
        if display_frame is not None:
            stframe.image(display_frame, channels="BGR")

        # Add a small sleep to prevent hogging the CPU
        time.sleep(0.01)

    cap.release()  # Release the camera
    cv2.destroyAllWindows()