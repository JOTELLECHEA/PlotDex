import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import os
import gradio as gr

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Define paths
model_path = "weights/CV_ResNet18_scratch_image_arxiv_232_raw_v1.pth"
labels_csv = "datasets/image_arxiv_232_raw_labels.csv"

# Define transformations 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model and labels
def load_model_and_labels():
    # Load label names
    df = pd.read_csv(labels_csv)
    label_names = df.columns[1:].tolist()  # Skip the image_path column
    num_classes = len(label_names)
    
    # Load model
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()  
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, label_names

# Prediction function for a single image
def predict_image(image, model, label_names):
    """Predict labels for a single image and return results with class names"""
    # Transform image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        confidence_scores = outputs.cpu().numpy()[0]
    
    # Format results as dictionary for Gradio
    results = {}
    for i, label in enumerate(label_names):
        results[label] = float(confidence_scores[i])
    
    return results

# Main prediction function that will be called by Gradio
def predict(input_image):
    if input_image is None:
        return {"No image provided": 0.0}
    
    # Load model and labels
    model, label_names = load_model_and_labels()
    
    # Convert from Gradio's format to PIL if needed
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image).convert('RGB')
    else:
        input_image = input_image.convert('RGB')
    
    # Get predictions
    predictions = predict_image(input_image, model, label_names)
    
    # Sort predictions by confidence (for display purposes)
    sorted_predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_predictions

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=None),  # Show all classes with confidence
    title="PlotDex",
    description="Upload a plot image and the model will analyze what type of plot it is.",
    examples=["datasets/examples/example-1.png","datasets/examples/example-2.png"] if os.path.exists("datasets/examples/example-1.png") else None
)

# Launch the app
if __name__ == "__main__":
    print(f"Starting Gradio app on device: {device}")
    demo.launch(share=True)  # share=True creates a public link