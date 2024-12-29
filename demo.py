import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision import models
import os
from PIL import Image

classes = ['O', 'R']

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None)  # Initialize the model architecture
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=2)  # Adjust for binary classification
model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=True))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to the demo folder
demo_folder = './demo'

# Predict images in the demo folder
for filename in os.listdir(demo_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image file types
        img_path = os.path.join(demo_folder, filename)
        image = Image.open(img_path).convert('RGB')  # Open and convert image
        image = transform(image).unsqueeze(0).to(device)  # Apply transformations and add batch dimension

        with torch.no_grad():
            output = model(image)  # Get model predictions
            _, predicted = torch.max(output, 1)  # Get the predicted class

        print(f'Image: {filename}, Predicted Class: {classes[predicted.item()]}')
