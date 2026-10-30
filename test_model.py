import torch
from torchvision import models, transforms
from PIL import Image


MODEL_PATH = "model/skin_model.pth"

classes = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

class_names = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevus",
    "vasc": "Vascular Lesion"
}


print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    7
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


image_path = input(
    "Enter image path: "
).strip().strip('"')


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


image = Image.open(
    image_path
).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)


with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )


predicted_class = classes[
    prediction.item()
]

predicted_name = class_names[
    predicted_class
]


print("\nPrediction:")
print(predicted_name)

print(
    "Confidence:",
    f"{confidence.item() * 100:.2f}%"
)