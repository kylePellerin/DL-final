from flask import Flask, request, send_file, jsonify
import torch
import io
from PIL import Image
from torchvision.utils import make_grid

from data import generator_DCGAN, noise_2d, countries_mapping, class_mapping, majors_mapping, show_generated_images 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = generator_DCGAN().to(device)
show_generated_images(generator)


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
    data = request.get_json()
    class_idx   = int(data["class_idx"])    # 0–3
    country_idx = int(data["country_idx"])  # 0–47
    major_idx   = int(data["major_idx"])    # 0–41

    with torch.no_grad():
        z = noise_2d(1).to(device)

        cond_class   = torch.zeros(1, 4,  device=device)
        cond_country = torch.zeros(1, 48, device=device)
        cond_major   = torch.zeros(1, 42, device=device)

        cond_class[0, class_idx]     = 1.0
        cond_country[0, country_idx] = 1.0
        cond_major[0, major_idx]     = 1.0

        conditions = torch.cat([cond_class, cond_country, cond_major], dim=1)

        fake = generator(z, conditions)          # (1, C, H, W)
        fake = (fake + 1) / 2.0                  # [-1,1] → [0,1]
        fake = fake.clamp(0, 1).cpu()

    # Convert tensor → PNG bytes
    grid = make_grid(fake, nrow=1, padding=0, normalize=False)
    ndarr = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
    img = Image.fromarray(ndarr)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True, port=5000)