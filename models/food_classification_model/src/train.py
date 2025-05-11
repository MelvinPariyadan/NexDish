import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from prepare import get_datasets_and_loaders
from model import load_model_architecture
from model import load_model_architecture


with open("models/food_classification_model/src/params.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device(config["device"] if torch.backends.mps.is_available() or config["device"] != "mps" else "cpu")

NUM_EPOCHS = config["train"]["num_epochs"]
LR = config["train"]["learning_rate"]
PATIENCE = config["train"]["patience"]
HIDDEN_DIM = config["train"]["hidden_dim"]

MODEL_PATH = config["output"]["model_checkpoint"]
MODEL_DIR = config["output"]["model_dir"]
PLOT_PATH = config["output"]["plot_file"]

train_loader, val_loader, test_loader, classes = get_datasets_and_loaders()
NUM_CLASSES = len(classes)

model = load_model_architecture(weights=ResNet50_Weights.IMAGENET1K_V1, device=device, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_accs, val_accs = [], []
train_losses, val_losses = [], []
best_val_acc = 0
patience_counter = 0
best_model_weights = None

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    correct_train = total_train = 0
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        correct_train += (preds.argmax(1) == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / total_train
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    model.eval()
    correct_val = total_val = 0
    val_loss_sum = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            val_loss = criterion(preds, labels)
            val_loss_sum += val_loss.item() * imgs.size(0)
            correct_val += (preds.argmax(1) == labels).sum().item()
            total_val += labels.size(0)

    val_loss = val_loss_sum / total_val
    val_acc = correct_val / total_val
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        best_model_weights = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

if best_model_weights:
    model.load_state_dict(best_model_weights)

model.eval()
correct_test = total_test = 0
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testing"):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        correct_test += (preds.argmax(1) == labels).sum().item()
        total_test += labels.size(0)

test_acc = correct_test / total_test
print(f"Test Accuracy: {test_acc:.4f}")

os.makedirs(MODEL_DIR, exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'classes': classes,
    'test_acc': test_acc,
    'best_val_acc': best_val_acc
}, MODEL_PATH)

epochs = range(1, len(train_accs) + 1)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, val_accs, label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid()

plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/{PLOT_PATH}")
plt.show()
