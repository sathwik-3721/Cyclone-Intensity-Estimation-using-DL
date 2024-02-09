# app.py

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np

# Model Architecture

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(784, 1),
    )
  def forward(self, x):
    return self.model(x)
  
def load_model():
    model = Model()
    model.load_state_dict(torch.load('C:\\Users\\DELL\\Desktop\\Codegnan\\Cyclone Intensity Estimation using Deep Learning\\cyclone_model', map_location=torch.device('cuda')))
    model.eval()
    return model

# Function to predict intensity in Knots
def predict_intensity(model, image):
    totensor = transforms.ToTensor()
    resize = transforms.Resize(size=(250, 250))
    img = resize(totensor(image)).unsqueeze(0)
    intensity = model(img).item()
    return intensity

# Streamlit App
def main():
    st.title("Cyclone Intensity Prediction App")

    # Load the model
    model = load_model()

    # User input for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button("Predict Intensity"):
            # Predict intensity
            intensity_knots = predict_intensity(model, image)
            st.success(f"Predicted Intensity: {intensity_knots} Knots")

if __name__ == '__main__':
    main()
