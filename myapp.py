import streamlit as st
import cv2
import numpy as np
from PIL import Image
from main import predict_image
from collections import Counter  # Tahmin istatistikleri için

st.title("Nesli Tükenmekte Olan Hayvan Tanıma Sistemi")
st.write("Kameranızı açarak bir hayvanı tanımlayın.")

run_camera = st.checkbox("Kamerayı Başlat")

if run_camera:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    prediction_counter = Counter()  # Tahminleri takip etmek için sayaç

    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.write("Kamera açılırken hata oluştu.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image)

        class_name, animal_class, confidence_score = predict_image(img_pil)

        # Tahminleri takip et
        prediction_counter[class_name] += 1

        # %95'in altında güven oranına sahip tahminleri gösterme
        if confidence_score < 0.95:
            continue

        label_text = f"{class_name} ({animal_class}) - {confidence_score*100:.2f}%"

        st.write(label_text)  # Ekrana yazdır
        cv2.putText(frame, label_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

    # Logları yazdır
    st.write("Tahmin Dağılımı:")
    st.write(dict(prediction_counter))
