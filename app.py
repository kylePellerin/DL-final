import torch
import io
import numpy as np
from PIL import Image
import torchvision.transforms as T
from flask import Flask, request, send_file, jsonify

# Import your mappings and model class
from data import countries_mapping, class_mapping, majors_mapping
from classifier_models import BowdoinClassifier

app = Flask(__name__, static_folder=".", static_url_path="")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize and Load Model correctly
# We instantiate the class first, THEN load the state_dict (weights)
model = BowdoinClassifier().to(device)
model.load_state_dict(torch.load("./Output_Classifier/bowdoin_classifier.pth", map_location=device))
model.eval()

# 2. Match the Training Transforms
# EfficientNet B3 usually expects 300x300, but use what you trained with.
# The normalization MUST match your training script exactly.
inference_transforms = T.Compose([
    T.Resize((224, 224)), 
    T.ToTensor(),
    T.Normalize((0.5173, 0.4501, 0.4103), (0.2840, 0.2643, 0.2671))
])

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/mappings")
def mappings():
    # Flips the dictionary so the frontend gets {0: "2024", 1: "2025"...}
    return jsonify({
        "class":   {v: k for k, v in class_mapping.items()},
        "country": {v: k for k, v in countries_mapping.items()},
        "major":   {v: k for k, v in majors_mapping.items()},
    })

@app.route("/analyze", methods=["POST"]) # Changed route name to match your HTML fetch('/analyze')
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Load Image
    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    
    # Preprocess
    img_tensor = inference_transforms(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits_class, logits_country, logits_major = model(img_tensor)

        # Process Class and Country (Softmax/Argmax)
        class_idx = logits_class.argmax(dim=1).item()
        country_idx = logits_country.argmax(dim=1).item()

        # Process Majors (Multi-label thresholding)
        major_probs = torch.sigmoid(logits_major[0])
        threshold = 0.25
        major_indices = (major_probs > threshold).nonzero(as_tuple=True)[0].tolist()
        
        # FALLBACK: If no major passes the threshold, take the single highest probability
        if not major_indices:
            major_indices = [major_probs.argmax().item()]

    # Map indices back to names using your imported mappings
    # Note: If your mapping is { "Computer Science": 0 }, we need to find the key by value
    inv_class = {v: k for k, v in class_mapping.items()}
    inv_country = {v: k for k, v in countries_mapping.items()}
    inv_major = {v: k for k, v in majors_mapping.items()}

    return jsonify({
        "class": inv_class.get(class_idx, "Unknown"),
        "country": inv_country.get(country_idx, "Unknown"),
        "major": ", ".join([inv_major.get(i, "Unknown") for i in major_indices]) if major_indices else "None"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)