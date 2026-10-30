import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


DATASET_PATH = "dataset"
MODEL_PATH = "model/skin_model_cnn.pth"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5

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
# CUSTOM CNN ARCHITECTURE
# =========================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
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


print("Loading metadata...")

metadata = pd.read_csv(os.path.join(DATASET_PATH, "HAM10000_metadata.csv"))
metadata["path"] = metadata["image_id"].apply(find_image)
metadata = metadata.dropna(subset=["path"])

print("Images available:", len(metadata))

train_df, val_df = train_test_split(
    metadata, test_size=0.2, random_state=42, stratify=metadata["dx"]
)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = SkinDataset(train_df, train_transform)
val_dataset = SkinDataset(val_df, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = SimpleCNN(num_classes=7).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nStarting training...\n")

for epoch in range(EPOCHS):

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
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    train_accuracy = (correct / total) * 100

    model.eval()
    val_total, val_correct = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            val_total += labels.size(0)
            val_correct += (predictions == labels).sum().item()

    val_accuracy = (val_correct / val_total) * 100

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {running_loss / len(train_loader):.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print("-" * 40)

os.makedirs("model", exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes,
    "class_names": class_names
}, MODEL_PATH)

print("\nTraining complete!")
print("Model saved to:", MODEL_PATH)