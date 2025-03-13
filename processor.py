import streamlit as st
import cv2
import numpy as np
from PIL import Image
from main import predict_image
import time

st.title("🐾 Nesli Tükenmekte Olan Hayvan Tanıma Sistemi")
st.write("📷 Kameranızı açarak bir hayvanı tanımlayın.")

# Kamera başlatma butonu
run_camera = st.checkbox("Kamerayı Başlat")

if run_camera:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows için CAP_DSHOW kullan
    stframe = st.empty()
    sttext = st.empty()  # Alt bilgi kutucuğu

    last_prediction_time = time.time()  # Son tahmin yapılan zamanı sakla

    while run_camera:
        ret, frame = cap.read()

        # Kameradan görüntü alınamıyorsa, sessizce devam et
        if not ret:
            time.sleep(0.5)
            continue

        # **Kamerayı güncelle (her döngüde çalışacak)**
        stframe.image(frame, channels="BGR")

        # **Her tahmini 0.3 saniyede bir yap (CPU yükünü azaltır)**
        if time.time() - last_prediction_time < 0.3:
            continue

        # **Tahmin için görüntüyü hazırla**
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image)

        class_name, category, confidence_score = predict_image(img_pil)
        last_prediction_time = time.time()  # Yeni tahmin zamanı kaydet

        # **Eğer Environment veya Human algılandıysa, kutucuğu kesin olarak temizle!**

        if class_name is None:
            sttext.empty()  # **Tahmin kutucuğunu tamamen temizle**
            continue  # Döngüyü devam ettir

        if class_name != "Environment" or class_name != "Human":

            label_text = f"**{class_name}**\n🟢 **Sınıfı:** {category}\n📊 **Güven Skoru:** {confidence_score * 100:.2f}%"

            # **Eğer yeni bir tahmin geldiyse, sadece o zaman ekrana yaz**
            if label_text != st.session_state.get("last_label", ""):
                st.session_state["last_label"] = label_text
                sttext.markdown(label_text)  # **Sadece geçerli hayvan tahmini yaz**



    cap.release()
    cv2.destroyAllWindows()
