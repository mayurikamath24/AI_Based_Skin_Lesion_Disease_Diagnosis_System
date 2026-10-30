import os
import sys
import copy
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, classification_report, confusion_matrix
)


DATASET_PATH = "dataset"
SPLIT_FILE = "model/splits.csv"

IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 42

# More epochs for the CNN because it learns from scratch
EPOCHS = {"efficientnet": 6, "mobilenet": 6, "resnet50": 6, "cnn": 10}

MODEL_FILES = {
    "efficientnet": "model/skin_model.pth",
    "mobilenet": "model/skin_model_mobilenet.pth",
    "resnet50": "model/skin_model_resnet50.pth",
    "cnn": "model/skin_model_cnn.pth",
}

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

DF_INDEX = classes.index("df")
MEL_INDEX = classes.index("mel")


# =========================
# DATA
# =========================

def find_image(image_id):
    for folder in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
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


def get_splits():
    """Split by LESION (not by image) so the same lesion never appears
    in both training and test data. Saved so every model uses the same split."""

    if os.path.exists(SPLIT_FILE):
        print("Using existing split file:", SPLIT_FILE)
        df = pd.read_csv(SPLIT_FILE)
        return (df[df["split"] == "train"],
                df[df["split"] == "val"],
                df[df["split"] == "test"])

    print("Creating new train/val/test split...")

    metadata = pd.read_csv(os.path.join(DATASET_PATH, "HAM10000_metadata.csv"))
    metadata["path"] = metadata["image_id"].apply(find_image)
    metadata = metadata.dropna(subset=["path"])

    print("Images available:", len(metadata))

    lesions = metadata.drop_duplicates("lesion_id")[["lesion_id", "dx"]]

    train_l, temp_l = train_test_split(
        lesions, test_size=0.30, stratify=lesions["dx"], random_state=SEED
    )
    val_l, test_l = train_test_split(
        temp_l, test_size=0.50, stratify=temp_l["dx"], random_state=SEED
    )

    metadata["split"] = "train"
    metadata.loc[metadata["lesion_id"].isin(val_l["lesion_id"]), "split"] = "val"
    metadata.loc[metadata["lesion_id"].isin(test_l["lesion_id"]), "split"] = "test"

    os.makedirs("model", exist_ok=True)
    metadata[["image_id", "lesion_id", "dx", "path", "split"]].to_csv(
        SPLIT_FILE, index=False
    )

    return (metadata[metadata["split"] == "train"],
            metadata[metadata["split"] == "val"],
            metadata[metadata["split"] == "test"])


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =========================
# MODELS (same structure as the app expects)
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
        return self.classifier(self.features(x))


def build_model(name):
    """Returns (model, list of new-head parameters)."""

    if name == "efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.features[-2:].parameters():      # unfreeze last 2 stages
            p.requires_grad = True
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
        head = list(model.classifier.parameters())

    elif name == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.features[-3:].parameters():      # unfreeze last 3 blocks
            p.requires_grad = True
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
        head = list(model.classifier.parameters())

    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():             # unfreeze last stage
            p.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, 7)
        head = list(model.fc.parameters())

    elif name == "cnn":
        model = SimpleCNN(num_classes=7)
        head = []

    else:
        raise ValueError("Unknown model: " + name)

    return model, head


# =========================
# EVALUATION HELPER
# =========================

def predict_all(model, loader, device):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            preds.extend(outputs.argmax(dim=1).cpu().tolist())
            labels_all.extend(labels.tolist())
    return labels_all, preds


# =========================
# TRAIN ONE MODEL
# =========================

def train_model(name, train_df, val_df, test_df, device):

    print("\n" + "=" * 60)
    print("TRAINING:", name)
    print("=" * 60)

    epochs = EPOCHS[name]

    train_loader = DataLoader(SkinDataset(train_df, train_transform),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(SkinDataset(val_df, eval_transform),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(SkinDataset(test_df, eval_transform),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model, head_params = build_model(name)
    model = model.to(device)

    # ----- class weights (square-root of inverse frequency) -----
    counts = train_df["dx"].value_counts()
    weights = []
    for c in classes:
        weights.append((len(train_df) / (len(classes) * counts[c])) ** 0.5)
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.mean()
    print("Class weights:", {c: round(w.item(), 2) for c, w in zip(classes, weights)})

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    # ----- optimizer: small lr for pretrained layers, larger for new head -----
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_ids]

    if name == "cnn":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": 1e-4},
            {"params": head_params, "lr": 1e-3},
        ])

    best_f1 = -1
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):

        start = time.time()
        model.train()
        total, correct, running_loss = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        val_labels, val_preds = predict_all(model, val_loader, device)

        val_acc = sum(int(a == b) for a, b in zip(val_labels, val_preds)) / len(val_labels) * 100
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        recalls = recall_score(val_labels, val_preds, labels=list(range(7)),
                               average=None, zero_division=0)

        print(f"\nEpoch {epoch + 1}/{epochs}  ({time.time() - start:.0f}s)")
        print(f"  Train loss: {running_loss / len(train_loader):.4f} | "
              f"Train acc: {correct / total * 100:.2f}%")
        print(f"  Val acc: {val_acc:.2f}% | Val macro-F1: {val_f1:.3f}")
        print(f"  Val recall -> melanoma: {recalls[MEL_INDEX] * 100:.0f}% | "
              f"dermatofibroma: {recalls[DF_INDEX] * 100:.0f}%")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            print("  * best so far (saved)")

    # ----- save best epoch -----
    os.makedirs("model", exist_ok=True)

    torch.save({
        "model_state_dict": best_state,
        "classes": classes,
        "class_names": class_names
    }, MODEL_FILES[name])

    print(f"\nSaved best epoch ({best_epoch}, val macro-F1 {best_f1:.3f}) to {MODEL_FILES[name]}")

    # ----- final report on the untouched TEST set -----
    model.load_state_dict(best_state)
    test_labels, test_preds = predict_all(model, test_loader, device)

    print("\nTEST SET RESULTS for", name)
    print(classification_report(
        test_labels, test_preds,
        labels=list(range(7)),
        target_names=[class_names[c] for c in classes],
        zero_division=0
    ))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(test_labels, test_preds, labels=list(range(7))))


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    names = sys.argv[1:] if len(sys.argv) > 1 else list(MODEL_FILES.keys())

    for n in names:
        if n not in MODEL_FILES:
            print("Unknown model:", n, "- choose from", list(MODEL_FILES.keys()))
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_df, val_df, test_df = get_splits()

    print("\nImages -> train:", len(train_df), "| val:", len(val_df), "| test:", len(test_df))
    print("Test set class counts:")
    print(test_df["dx"].value_counts())

    for n in names:
        train_model(n, train_df, val_df, test_df, device)

    print("\nAll done.")