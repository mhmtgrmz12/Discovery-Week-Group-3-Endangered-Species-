# Endangered Animal Recognition System

A real-time computer vision application that identifies and tracks endangered animal species using AI. The system classifies animals according to their conservation status, logs detections, and provides visualization tools for analysis.

## 🌟 Features

- **Real-time Animal Detection**: Uses a pre-trained Keras model to identify animals with >95% confidence
- **Conservation Status Classification**: Groups animals into conservation categories (EN, VU, NT, LC)
- **Smart Logging System**: Records detections with a cooldown period to prevent duplicates
- **Interactive Dashboard**: Visualizes detection trends and conservation category distribution
- **Species Information**: Provides details about detected animals and their conservation status

## 📋 Requirements

- Python 3.11.9
- Streamlit
- TensorFlow/Keras
- OpenCV
- Pillow
- Pandas
- Plotly
- Additional dependencies listed in `requirements.txt`

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/endangered-animal-recognition.git
   cd endangered-animal-recognition
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install system dependencies:
   ```bash
   apt-get update && apt-get install -y libgl1
   ```

## 🏃‍♂️ Running the Application

1. Start the Streamlit application:
   ```bash
   streamlit run main.py
   ```

2. The application will open in your default web browser. If not, navigate to `http://localhost:8501`.

## 📁 Project Structure

```
endangered-animal-recognition/
├── app.py                 # Main application with camera feed and detection UI
├── logs.py                # Detection logs visualization dashboard
├── main.py                # Streamlit application entry point
├── modular.py             # Model loading and prediction functions
├── requirements.txt       # Python dependencies
├── packages.txt           # System dependencies
├── runtime.txt            # Python version specification
├── labels.txt             # Model class labels
├── keras_model.h5         # Pre-trained model (not included in repository)
├── .streamlit/
│   └── pages.toml         # Navigation configuration
└── logs/
    └── detection_logs.json # Detection history log file
```

## 🔄 How It Works

1. **Image Capture**: The system captures video frames from your webcam
2. **Image Processing**: Each frame is processed and prepared for the model
3. **AI Classification**: The Keras model predicts the animal species in the frame
4. **Result Filtering**: Only high-confidence detections (>95%) are processed
5. **Status Lookup**: The system looks up conservation status from its database
6. **Logging**: New detections are logged with timestamp and confidence data
7. **Visualization**: The logs dashboard provides analysis of detection history

## 🧠 Model Details

The system uses a Keras model trained to recognize various endangered animal species. The model classifies animals into the following conservation categories:

- **EN(G1)**: Critically endangered species (Vaquita, Saola, Eastern Lowland Gorilla, etc.)
- **EN(G2)**: Endangered species (Black-footed Ferret, Sea Turtle, Red Panda, etc.)
- **VU(G3)**: Vulnerable species (Black Spider Monkey, Lion, Greater One-Horned Rhino, etc.)
- **NT(G4)**: Near-threatened species (Mountain Plover, Yellowfin Tuna, Jaguar, etc.)
- **LC(G5)**: Least concern species (Beaver, Tree Kangaroo, Macaw, etc.)

## 📊 Analytics Dashboard

The system includes a comprehensive analytics dashboard that:
- Displays summary metrics (total detections, unique species, avg. confidence)
- Shows detection trends over time
- Visualizes the distribution of detections by conservation category
- Allows filtering by date range, category, and confidence score
- Provides downloadable detection logs in CSV format

## 🔧 Configuration

The application automatically creates necessary directories and log files if they don't exist. The main configuration files include:

- **labels.txt**: Contains the mapping between model output indices and animal names
- **database/endangered.json**: Contains detailed information about endangered species (scientific names, conservation status)

## 📝 Notes

- The system implements a 30-second cooldown between detections of the same species to prevent duplicate logs
- Environment and human detections are automatically filtered out
- Camera settings are optimized for performance with reduced latency

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

