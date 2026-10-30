import os
import sys

# Let Python find the project root, so "backend.xxx" imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
import uuid
import threading
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# =========================
# PROJECT FOLDERS
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_FOLDER = os.path.join(BASE_DIR, "frontend")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "model")
RAG_DIR = os.path.join(BASE_DIR, "rag", "rag")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# FLASK
# =========================

app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path="")
CORS(app)


# =========================
# CLASSES
# =========================

classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

class_names = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevus",
    "vasc": "Vascular Lesion"
}


# Warn when fewer than this many of the 4 models agree on the class.
AGREEMENT_WARN_BELOW = 3


# =========================
# CUSTOM CNN ARCHITECTURE
# (must match train_cnn.py exactly)
# =========================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_efficientnet():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    checkpoint = torch.load(os.path.join(MODEL_DIR, "skin_model.pth"), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_mobilenet():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    checkpoint = torch.load(os.path.join(MODEL_DIR, "skin_model_mobilenet.pth"), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_resnet50():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    checkpoint = torch.load(os.path.join(MODEL_DIR, "skin_model_resnet50.pth"), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_cnn():
    model = SimpleCNN(num_classes=7)
    checkpoint = torch.load(os.path.join(MODEL_DIR, "skin_model_cnn.pth"), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


print("Loading all 4 models...")

MODELS = {
    "EfficientNet-B0": load_efficientnet(),
    "MobileNetV2": load_mobilenet(),
    "ResNet50": load_resnet50(),
    "Custom CNN": load_cnn(),
}

for name, m in MODELS.items():
    m.eval()
    print(f"  {name} loaded.")

print("All models loaded successfully!")


# =========================
# RAG + KNOWLEDGE GRAPH (explanations)
# Optional: if it fails to load, classification still works.
# =========================

sys.path.insert(0, RAG_DIR)

try:
    print("Loading explanation system (RAG + knowledge graph)...")
    from rag import generate_answer as rag_generate_answer
    RAG_READY = True
    print("Explanation system loaded.")
except Exception as e:
    RAG_READY = False
    print("WARNING: explanation system unavailable:", e)

rag_lock = threading.Lock()

try:
    from backend.kg import get_subgraph, get_condition_context
    KG_READY = True
except Exception as e:
    KG_READY = False
    print("WARNING: knowledge graph unavailable:", e)


# =========================
# IMAGE TRANSFORMATION
# (same for all 4 models)
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =========================
# FRONTEND
# =========================

@app.route("/")
def home():
    return send_from_directory(FRONTEND_FOLDER, "index.html")


@app.route("/<path:path>")
def frontend_files(path):
    return send_from_directory(FRONTEND_FOLDER, path)


# =========================
# IMAGE UPLOAD + MULTI-MODEL ANALYSIS
# =========================

@app.route("/upload", methods=["POST"])
def upload_image():

    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"}), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({"success": False, "message": "No image selected"}), 400

    safe_name = uuid.uuid4().hex + "_" + secure_filename(image.filename)
    filepath = os.path.join(UPLOAD_FOLDER, safe_name)
    image.save(filepath)

    try:
        img = Image.open(filepath).convert("RGB")
        img = transform(img)
        img = img.unsqueeze(0)

        # =========================
        # RUN EVERY MODEL
        # =========================

        all_results = {}

        for model_name, model in MODELS.items():

            with torch.no_grad():
                output = model(img)
                probabilities = torch.softmax(output, dim=1)
                confidence, prediction = torch.max(probabilities, dim=1)

            predicted_class = classes[prediction.item()]

            all_results[model_name] = {
                "class": predicted_class,
                "class_name": class_names[predicted_class],
                "confidence": round(confidence.item() * 100, 2)
            }

        # =========================
        # PICK THE MOST CONFIDENT MODEL
        # =========================

        winner_name = max(
            all_results,
            key=lambda name: all_results[name]["confidence"]
        )

        winner = all_results[winner_name]
        confidence_value = winner["confidence"]

        # How many of the models picked the same class as the winner
        agree_count = sum(
            1 for r in all_results.values()
            if r["class"] == winner["class"]
        )

        agreement = {
            "count": agree_count,
            "total": len(all_results),
            "warn_below": AGREEMENT_WARN_BELOW
        }

        # =========================
        # LOW CONFIDENCE HANDLING
        # =========================

        if confidence_value < 50:

            return jsonify({
                "success": True,
                "filename": safe_name,
                "prediction": "Uncertain",
                "class": winner["class"],
                "class_name": winner["class_name"],
                "confidence": confidence_value,
                "responsible_model": winner_name,
                "all_models": all_results,
                "agreement": agreement,
                "message": "The model is not confident enough to provide a reliable prediction."
            })

        return jsonify({
            "success": True,
            "filename": safe_name,
            "prediction": winner["class_name"],
            "class": winner["class"],
            "class_name": winner["class_name"],
            "confidence": confidence_value,
            "responsible_model": winner_name,
            "all_models": all_results,
            "agreement": agreement,
            "message": "Image analyzed successfully."
        })

    except Exception as e:
        print("Analysis error:", str(e))
        return jsonify({
            "success": False,
            "message": "Unable to analyze the image.",
            "error": str(e)
        }), 500


# =========================
# EXPLANATION (RAG + KNOWLEDGE GRAPH)
# Separate endpoint so the prediction shows immediately and the
# slower LLM answer follows.
# =========================

@app.route("/explain", methods=["POST"])
def explain():

    data = request.get_json(silent=True) or {}

    class_code = data.get("class")
    uncertain = bool(data.get("uncertain", False))

    # Optional: how many of the 4 models agreed (validated integer only)
    agree = data.get("agree")
    extra_note = None
    if isinstance(agree, int) and not isinstance(agree, bool) and 0 <= agree <= len(MODELS):
        extra_note = f"Only {agree} of {len(MODELS)} AI models agreed on this result."
        if agree >= len(MODELS):
            extra_note = None

    # Only accept a known class code - never free text from the client.
    if class_code not in classes:
        return jsonify({"success": False, "message": "Unknown class."}), 400

    if uncertain:
        question = (
            "The AI result was uncertain. Explain in simple terms what "
            "this closest guess is, why it should not be relied on, "
            "and what the person should do next."
        )
    else:
        question = (
            "Explain this possible skin condition in simple terms "
            "and what the person should do next."
        )

    # 1) Preferred: language model + retrieved text + knowledge graph
    if RAG_READY:
        try:
            with rag_lock:
                answer = rag_generate_answer(
                    question,
                    condition=class_code,
                    uncertain=uncertain,
                    extra_note=extra_note
                )

            return jsonify({"success": True, "explanation": answer})

        except Exception as e:
            print("Explanation error (using fallback):", repr(e))

    # 2) Fallback: plain text built from the curated knowledge graph only,
    #    so an explanation still appears if the language model fails.
    if KG_READY:
        text = get_condition_context(class_code, uncertain=uncertain)

        if extra_note:
            text += "\n" + extra_note

        text += (
            "\nThis is only an AI estimate, not a diagnosis. "
            "Please see a qualified healthcare professional."
        )

        return jsonify({
            "success": True,
            "explanation": text,
            "fallback": True
        })

    return jsonify({
        "success": False,
        "message": "Explanation system is not available."
    }), 503


# =========================
# KNOWLEDGE GRAPH (for the visual)
# =========================

@app.route("/graph/<code>")
def graph(code):

    if code not in classes:
        return jsonify({"success": False, "message": "Unknown class."}), 400

    if not KG_READY:
        return jsonify({
            "success": False,
            "message": "Knowledge graph is not available."
        }), 503

    return jsonify({"success": True, "graph": get_subgraph(code)})


# =========================
# START SERVER
# =========================

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)