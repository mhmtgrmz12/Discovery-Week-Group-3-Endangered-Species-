from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Modeli yükle
model = load_model("keras_model.h5", compile=False)

# Etiketleri yükle
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

def predict_image(image):
    """Verilen görüntü için modelden tahmin döndürür."""
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    probabilities = prediction[0] / np.sum(prediction[0])  # Normalizasyon

    index = np.argmax(probabilities)
    confidence_score = probabilities[index]
    class_name = class_names[index]

    # **Eğer environment algılanırsa hiçbir şey döndürme**
    if class_name == "Environment":
        return "Environment", "N/A", 0  # Ana kodda görmezden gelinecek

    # **Eğer hayvan algılanırsa ama doğruluk %90'ın altındaysa hiçbir şey döndürme**
    if confidence_score < 0.90:
        return "Unknown", "N/A", 0  # Ana kodda görmezden gelinecek

    return class_name, "Hayvan", confidence_score
