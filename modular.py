from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import re  # Sayıları temizlemek için regex kullanacağız

# Modeli yükle
model = load_model("keras_model.h5", compile=False)


# **labels.txt içinden boşlukları ve başındaki sayıları temizle**
def clean_label(line):
    """Boş satırları ve başındaki sayıları temizleyen fonksiyon"""
    line = line.strip()
    line = re.sub(r'^\d+\s+', '', line)  # Başındaki sayıları sil
    return line


with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [clean_label(line) for line in f.readlines() if line.strip()]  # Boş satırları temizle

# Nesli tükenmekte olan hayvanları sınıflarına göre gruplama
animal_classes = {
    "EN(G1)": ["Vaquita", "Pseudoryx nghetinhensis saola", "Eastern Lowland Gorilla",
               "Bornean Orangutan", "Black Rhino", "Amur Leopard", "African forest elephant"],
    "EN(G2)": ["Black-footed Ferret", "Sea Turtle", "Red Panda", "Monarch Butterfly",
               "Humphead Wrasse", "Whale Shark", "African Wild Dog", "Sea Lions", "Chimpanzee"],
    "VU(G3)": ["Black Spider Monkey", "Lion", "Greater One-Horned Rhino", "Dugong",
               "Hippopotamus", "Olive Ridley Turtle"],
    "NT(G4)": ["Mountain Plover", "Yellowfin Tuna", "Greater Sage-Grouse", "Plains Bison", "Jaguar"],
    "LC(G5)": ["Beaver", "Tree Kangaroo", "Macaw", "Swift Fox", "Arctic Wolf", "Arctic Fox"]
}


def get_animal_class(animal_name):
    """Hayvanın nesli tükenme sınıfını döndürür."""
    for category, animals in animal_classes.items():
        if animal_name in animals:
            return category
    return "Unknown"


def predict_image(image):
    """Verilen görüntü için model tahmini döndürür."""
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    probabilities = prediction[0] / np.sum(prediction[0])

    index = np.argmax(probabilities)
    confidence_score = probabilities[index]

    # **labels.txt içinden doğru etiketi al**
    if index < len(class_names):
        class_name = class_names[index].strip()
    else:
        class_name = "Unknown"

    # **Eğer Environment veya Human algılandıysa, kesin olarak `None` döndür!**
    if class_name in ["Environment", "Human"]:
        return None, None, None

    # Eğer hayvan algılanırsa ama doğruluk %90'ın altındaysa hiçbir şey döndürme
    if confidence_score < 0.90:
        return None, None, None

    # Hayvanın nesli tükenme sınıfını al
    animal_class = get_animal_class(class_name)

    return class_name, animal_class, confidence_score