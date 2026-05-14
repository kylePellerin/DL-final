from flask import Flask, request, send_file, jsonify
# from shapely import transform
import torch
import io
import numpy as np
from PIL import Image
from torchvision.utils import make_grid

from data import countries_mapping, class_mapping, majors_mapping
from models import generator_DCGAN
from classifier_models import BowdoinClassifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = generator_DCGAN().to(device)


app = Flask(__name__, static_folder=".", static_url_path="")


@app.route("/mappings")
def mappings():
    return jsonify({
        "class":   {v: k for k, v in class_mapping.items()},
        "country": {v: k for k, v in countries_mapping.items()},
        "major":   {v: k for k, v in majors_mapping.items()},
    })


def index():
    return send_file("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    pass

@app.route("/findAttributes", methods=["POST"])
def findAttributes():
    data = request.files["image"].read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize((800, 600)) # Resize to match training data
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 # (C, H, W) in [0,1]
    bowdoin_classifier = torch.load("bowdoin_classifier.pth", map_location=device)
    
    bowdoin_classifier.eval()
    class_year, country, major = bowdoin_classifier.forward(img_tensor.unsqueeze(0).to(device)) # Add batch dimension
    return jsonify({
        "class year probability distribution": class_year,
        "possible study abroad countries": country,
        "possible majors": major
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)