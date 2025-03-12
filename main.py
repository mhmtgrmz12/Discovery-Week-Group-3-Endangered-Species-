from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Modeli yükle
model = load_model("keras_model.h5", compile=False)

# Etiketleri yükle
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Hayvan sınıfları sözlüğü
animal_classes = {
    "EN(G1)": [
        "Vaquita",
        "Pseudoryx nghetinhensis saola",
        "Eastern Lowland Gorilla",
        "Bornean Orangutan",
        "Black Rhino",
        "Amur Leopard",
        "African forest elephant"
    ],
    "EN(G2)": [
        "Black-footed Ferret",
        "Sea Turtle",
        "Red Panda",
        "Monarch Butterfly",
        "Humphead Wrasse",
        "Whale Shark",
        "African Wild Dog",
        "Sea Lions",
        "Chimpanzee"
    ],
    "VU(G3)": [
        "Black Spider Monkey",
        "Lion",
        "Greater One-Horned Rhino",
        "Dugong",
        "Hippopotamus",
        "Olive Ridley Turtle"
    ],
    "NT(G4)": [
        "Mountain Plover",
        "Beluga",
        "Yellowfin Tuna",
        "Greater Sage-Grouse",
        "Plains Bison",
        "Jaguar"
    ],
    "LC(G5)": [
        "Beaver",
        "Tree Kangaroo",
        "Macaw",
        "Swift Fox",
        "Arctic Wolf",
        "Arctic Fox"
    ]
}


def get_animal_class(animal_name):
    """Hayvanın hangi sınıfa ait olduğunu döndürür."""
    animal_name = animal_name.strip().lower()  # Boşlukları ve büyük harfleri düzelt

    print(f"🔥 Aranan hayvan: {animal_name}")  # Debugging

    for class_name, animals in animal_classes.items():
        print(f"🟢 Kontrol Edilen Sınıf: {class_name}")  # Debugging
        for animal in animals:
            print(f"   🔍 Karşılaştırılıyor: {animal.strip().lower()} == {animal_name}")  # Debugging
            if animal.strip().lower() == animal_name:
                print(f"✅ Eşleşme bulundu: {animal} -> {class_name}")  # Debugging
                return class_name

    print("❌ Hiçbir eşleşme bulunamadı!")  # Debugging
    return "Bilinmeyen Sınıf"




def predict_image(image):
    """Verilen görüntü için modelden tahmin döndürür."""
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)

    # Sayıyı kaldırarak sadece hayvan ismini al
    class_name = " ".join(class_names[index].strip().split(" ")[1:])
    confidence_score = prediction[0][index]

    # Hayvanın soyunu bul
    animal_class = get_animal_class(class_name)

    return class_name, animal_class, confidence_score
