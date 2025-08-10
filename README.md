# Plant Disease Detector Web App 🌱

A machine learning-powered web application that identifies plant diseases from leaf images. Built with Streamlit and TensorFlow, this app can detect 38 different plant diseases and healthy conditions across 14 crop types.

## Features

- Image Upload & Analysis: Upload leaf images for instant disease detection.
- Disease Identification: Detects 38 different diseases and healthy leaf conditions.
- Detailed Descriptions: Provides comprehensive information about each detected disease.
- Prevention Tips: Offers practical prevention measures for identified diseases.
- User-Friendly Interface: Clean, responsive web interface built with Streamlit.

## Crops Covered
- Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

## Disease Types
- Fungal diseases (Apple Scab, Powdery Mildew, Late Blight, etc.)
- Bacterial diseases (Bacterial Spot, Leaf Scorch, etc.)
- Viral diseases (Tomato Mosaic Virus, Yellow Leaf Curl Virus)
- Pest damage (Spider Mites)
- Healthy leaf identification

## Technology Stack
- Frontend: Streamlit
- Backend: Python
- Machine Learning: TensorFlow/Keras with Swin Transformer
- Image Processing: PIL (Python Imaging Library)
- Model Architecture: Swin Transformer (swin.h5)

## Installation & Setup
### Prerequisites
- Python 3.7 or higher
- pip package manager

1. Clone the repository
```bash
git clone https://github.com/yourusername/plant-disease-detector.git
cd plant-disease-detector
```
2. Install Dependencies
```bash
pip install streamlit tensorflow pillow numpy absl-py
```
3. Download the Model
You'll need to place the trained model file (swin.h5) in the trained_model/ directory:
```bash
plant-disease-detector/
├── trained_model/
│   └── swin.h5
├── main.py
├── class_indices.json
├── disease_descriptions.json
├── disease_preventions.json
└── config.toml
```
4. Run the Application
```bash
streamlit run main.py
```
The app will be available at http://localhost:8501 

Usage 📖

## Upload Image 
1. Click "Upload an image" and select a clear photo of a plant leaf
2. Predict: Click the "Predict" button to analyze the image
3. View Results: The app will display:
 - The predicted disease/condition
 - Description of the disease
 - Prevention measures

