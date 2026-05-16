import torch
import io
import numpy as np
from PIL import Image
import torchvision.transforms as T
from flask import Flask, request, send_file, jsonify
from collections import OrderedDict
import torch.nn.functional as F
from data import countries_mapping, class_mapping, majors_mapping
from models import DCGAN_gen
from classifier_models import BowdoinClassifier

app = Flask(__name__, static_folder=".", static_url_path="")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BowdoinClassifier().to(device)
model.load_state_dict(torch.load("./Output_Classifier/bowdoin_classifier.pth", map_location=device))
model.eval()

generator = DCGAN_gen().to(device)

state_dict = torch.load("celeba_generator_final.pth", map_location=device)
#create a new state_dict that only contains the keys that match the generator's architecture
try:
    generator.load_state_dict(state_dict, strict=False)
    print("✓ GAN weights loaded successfully.")
except Exception as e:
    print(f"Error loading GAN: {e}")

def get_major_multihot(major_idx, num_majors=42):
    vec = torch.zeros(num_majors)
    vec[major_idx] = 1.0
    return vec


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    class_idx = data.get('class_idx')
    country_idx = data.get('country_idx')
    major_idx = data.get('major_idx')

    noise = torch.randn(1, 100, 1, 1).to(device)
    
    #convert to one-hot vectors
    c_label = F.one_hot(torch.tensor([class_idx]), num_classes=4).float().to(device)
    co_label = F.one_hot(torch.tensor([country_idx]), num_classes=48).float().to(device)
    m_label = get_major_multihot(major_idx, num_majors=42).unsqueeze(0).to(device)
    combined_labels = torch.cat([c_label, co_label, m_label], dim=1)

    with torch.no_grad():
        fake_image = generator(noise, combined_labels)
        
        img_tensor = fake_image.squeeze(0).cpu()
        img_tensor = (img_tensor + 1.0) / 2.0
        img_tensor = torch.clamp(img_tensor, 0, 1)

        img_pil = T.ToPILImage()(img_tensor)
        img_io = io.BytesIO()
        img_pil.save(img_io, 'PNG')
        img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

#match the same transforms used during training for the classifier
inference_transforms = T.Compose([
    T.Resize((224, 224)), 
    T.ToTensor(),
    T.Normalize((0.5173, 0.4501, 0.4103), (0.2840, 0.2643, 0.2671))
])

@app.route("/mappings")
def mappings():
    return jsonify({
        "class":   {v: k for k, v in class_mapping.items()},
        "country": {v: k for k, v in countries_mapping.items()},
        "major":   {v: k for k, v in majors_mapping.items()},
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    
    img_tensor = inference_transforms(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits_class, logits_country, logits_major = model(img_tensor)

        #process class and country
        if '2028' in class_mapping:
            idx_2028 = class_mapping['2028']
            logits_class[0][idx_2028] -= 7.0
        class_idx = logits_class.argmax(dim=1).item()
        country_idx = logits_country.argmax(dim=1).item()

        #process major 
        major_probs = torch.sigmoid(logits_major[0])
        threshold = 0.25
        major_indices = (major_probs > threshold).nonzero(as_tuple=True)[0].tolist()
        
        if not major_indices:
            major_indices = [major_probs.argmax().item()]
        
        major_indices = major_indices[:2]

    #map indices back to labels
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