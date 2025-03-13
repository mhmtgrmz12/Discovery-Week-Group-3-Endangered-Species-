import streamlit as st
import cv2
import numpy as np
from PIL import Image
from main import predict_image
import time

st.title("Nesli Tükenmekte Olan Hayvan Tanıma Sistemi")
st.write("Kameranız sürekli açık olacak. Sadece hayvan algılandığında ve doğruluk %90 üzerindeyse çıktı verilecek.")

# Kamera açma butonu
run_camera = st.checkbox("Kamerayı Başlat")

if run_camera:
    cap = cv2.VideoCapture(0)  # Kamerayı başlat
    stframe = st.empty()

    if not cap.isOpened():  # Kamera açılamadıysa hata ver
        st.error("Kamera açılamadı! Başka bir program kullanıyor olabilir.")
        run_camera = False  # Döngüyü bitir

    while run_camera:
        ret, frame = cap.read()

        # Eğer kameradan görüntü alınamıyorsa, döngüyü bir süre beklet ve tekrar dene
        if not ret:
            st.warning("Kameradan görüntü alınamıyor! Bağlantıyı kontrol edin.")
            time.sleep(0.5)  # Bekleyerek CPU yükünü azalt
            continue  # Döngüyü devam ettir

        # Görüntüyü renk formatına çevir
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image)

        class_name, category, confidence_score = predict_image(img_pil)

        # **Eğer environment algılanırsa hiçbir şey yazdırmayacağız**
        if class_name == "Environment":
            continue  # Döngüyü devam ettir, ekrana yazma

        # **Eğer hayvan algılanırsa ama doğruluk %90'ın altındaysa hiçbir şey göstermeyeceğiz**
        if confidence_score < 0.90:
            continue

        label_text = f"{class_name} - {confidence_score * 100:.2f}%"

        # Eğer yeni bir tahmin geldiyse, sadece o zaman ekrana yaz
        if label_text != st.session_state.get("last_label", ""):
            st.session_state["last_label"] = label_text
            st.write(label_text)

        cv2.putText(frame, label_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()  # Kamerayı serbest bırak
    cv2.destroyAllWindows()
