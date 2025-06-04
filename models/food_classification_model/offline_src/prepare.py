# prepare.py
import os
import random
import shutil
import yaml
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, random_split
from transform import get_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(PROJECT_DIR, 'params.yaml')

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config["train"]["batch_size"]
VAL_SPLIT = config["dataset"]["val_split"]
BACKEND_SAMPLES = config["dataset"]["backend_samples"]

DATA_DIR = 'datasets/food-101'
IM_DIR = os.path.join(DATA_DIR, "images")

classes = sorted([d for d in os.listdir(IM_DIR) if os.path.isdir(os.path.join(IM_DIR, d))])
class2idx = {c: i for i, c in enumerate(classes)}
idx2class = {i: c for c, i in class2idx.items()}

class Food101Dataset(Dataset):
    def __init__(self, items=None, transform=None):
        self.transform = transform
        self.items = items or []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def get_datasets_and_loaders(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True):
    train_items, test_items_full = [], []
    for cls in classes:
        cls_dir = os.path.join(IM_DIR, cls)
        images = [img for img in os.listdir(cls_dir) if img.endswith(".jpg")]
        random.shuffle(images)
        split_idx = int(0.8 * len(images))
        cls_train = images[:split_idx]
        cls_test = images[split_idx:]

        for img in cls_train:
            path = os.path.join(cls_dir, img)
            train_items.append((path, class2idx[cls]))

        for img in cls_test:
            path = os.path.join(cls_dir, img)
            test_items_full.append((path, class2idx[cls]))

    val_size = int(len(test_items_full) * VAL_SPLIT)
    test_size = len(test_items_full) - val_size
    val_items, test_items = random_split(test_items_full, [val_size, test_size])

    train_tfms, val_tfms = get_transforms()
    train_ds = Food101Dataset(items=train_items, transform=train_tfms)
    val_ds = Food101Dataset(items=val_items, transform=val_tfms)
    test_ds = Food101Dataset(items=test_items, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader, test_loader, classes
