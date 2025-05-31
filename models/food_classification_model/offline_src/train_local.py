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
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()
wandb_key = os.getenv("WANDB_API_KEY")
import wandb
wandb.login(wandb_key)
current_time = datetime.now()
tstamp = current_time.strftime("%Y-%m-%d_%H:%M:%S")


# Local directories
PROJECT_DIR = os.path.dirname(__file__) #make paths absolute
DATA_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", "..", "datasets/food-101"))
CONFIG_PATH = os.path.join(PROJECT_DIR, 'params.yaml')
MODEL_DIR = os.path.join(PROJECT_DIR, '../checkpoints')
MODEL_PATH = os.path.join(MODEL_DIR, f'checkpoint_{tstamp}.pth')
PLOT_PATH = os.path.join(MODEL_DIR, f'resnet_50_model_test_{tstamp}.png')
RESULTS_PATH = os.path.join(MODEL_DIR, f'accuracy_checkpoint_{tstamp}.txt')

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

print("Using config:", config)


PIN_MEMORY = True # Improves GPU perf. by enabling faster CPU->GPU loads
NUM_WORKERS = 4

# check GPU availability
if torch.cuda.is_available():
    print("Using CUDA for training.")
    device = "cuda"
elif torch.backends.mps.is_available():
    print("Using MPS for training.")
    device = "mps"
else:
    print("Using CPU for training.")
    device = "cpu"

 # Hyperparameters
BATCH_SIZE = config["train"]["batch_size"]
NUM_EPOCHS = config["train"]["num_epochs"]
LR = config["train"]["learning_rate"]
WEIGHT_DECAY = config["train"]["weight_decay"]
PATIENCE = config["train"]["patience"]
HIDDEN_DIM = config["train"]["hidden_dim"]

train_loader, val_loader, test_loader, classes = get_datasets_and_loaders(BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
NUM_CLASSES = len(classes)

model = load_model_architecture(weights=ResNet50_Weights.IMAGENET1K_V1, device=device, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler() #use amp for faster (mixed precision) training

train_accs, val_accs = [], []
train_losses, val_losses = [], []
best_val_acc = 0
patience_counter = 0
best_model_weights = None

total_runs = 5
for run in range(total_runs):
    # init wandb run
    wandb.init(
        project="NexDish",
        group="tuning",
        name=f"train_run-{run}_{tstamp}",
        config={
            "learning_rate": LR,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "model": "resnet50"
        })
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        correct_train = total_train = 0
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast(device_type="cuda"):
                preds = model(imgs)
                loss = criterion(preds, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
        wandb.log({"Training Loss": train_loss, 
                    "Validation Loss": val_loss,
                    "Validation Accuracy": val_acc, 
                    "Training Accuracy": train_acc, 
                    "Epoch": epoch})
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
    wandb.finish()

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

with open(RESULTS_PATH, "w") as f:
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")

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
plt.savefig(f"{PLOT_PATH}")
plt.show()
