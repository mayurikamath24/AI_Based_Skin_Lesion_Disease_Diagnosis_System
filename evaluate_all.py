import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


DATASET_PATH = "dataset"
IMG_SIZE = 224
BATCH_SIZE = 16

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


def find_image(image_id):
    folders = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
    for folder in folders:
        path = os.path.join(DATASET_PATH, folder, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None


class SkinDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        label = classes.index(row["dx"])
        if self.transform:
            image = self.transform(image)
        return image, label


# =========================
# SAME CUSTOM CNN ARCHITECTURE AS train_cnn.py
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
    checkpoint = torch.load("model/skin_model.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_mobilenet():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    checkpoint = torch.load("model/skin_model_mobilenet.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_resnet50():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    checkpoint = torch.load("model/skin_model_resnet50.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_cnn():
    model = SimpleCNN(num_classes=7)
    checkpoint = torch.load("model/skin_model_cnn.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


MODEL_LOADERS = {
    "EfficientNet-B0": load_efficientnet,
    "MobileNetV2": load_mobilenet,
    "ResNet50": load_resnet50,
    "Custom CNN": load_cnn,
}


# =========================
# REBUILD THE SAME VALIDATION SPLIT
# (same seed/test_size as every train_*.py, so this is
#  data none of the models were trained on)
# =========================

print("Loading metadata...")

metadata = pd.read_csv(os.path.join(DATASET_PATH, "HAM10000_metadata.csv"))
metadata["path"] = metadata["image_id"].apply(find_image)
metadata = metadata.dropna(subset=["path"])

train_df, val_df = train_test_split(
    metadata, test_size=0.2, random_state=42, stratify=metadata["dx"]
)

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = SkinDataset(val_df, val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Validation images:", len(val_dataset))


# =========================
# EVALUATE EACH MODEL
# =========================

for model_name, loader_fn in MODEL_LOADERS.items():

    print("\n" + "=" * 60)
    print(f"Evaluating: {model_name}")
    print("=" * 60)

    model = loader_fn()
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            all_preds.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    print("\nClassification report (per class):\n")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=[class_names[c] for c in classes],
        zero_division=0
    ))

    print("Confusion matrix (rows = true class, columns = predicted class):")
    print("Order:", [class_names[c] for c in classes])
    print(confusion_matrix(all_labels, all_preds))

print("\nDone evaluating all models.")