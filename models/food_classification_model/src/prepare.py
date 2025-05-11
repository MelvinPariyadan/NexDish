# prepare.py
import os
import random
import shutil
import yaml
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, random_split
from transform import get_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

with open("models/food_classification_model/src/params.yaml") as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config["train"]["batch_size"]
VAL_SPLIT = config["dataset"]["val_split"]
BACKEND_SAMPLES = config["dataset"]["backend_samples"]

BASE_DIR = config["dataset"]["base_dir"]
IM_DIR = config["dataset"]["image_dir"]
META_DIR = config["dataset"]["meta_dir"]
TRAIN_FILE = config["dataset"]["train_file"]
TEST_FILE = config["dataset"]["test_file"]
CLASSES_FILE = config["dataset"]["classes_file"]
BACKEND_DIR = config["dataset"]["backend_dir"]

os.makedirs(BACKEND_DIR, exist_ok=True)

with open(CLASSES_FILE) as f:
    classes = [line.strip() for line in f if line.strip()]
class2idx = {c: i for i, c in enumerate(classes)}
idx2class = {i: c for c, i in class2idx.items()}

class Food101Dataset(Dataset):
    def __init__(self, split_txt=None, items=None, transform=None):
        self.transform = transform
        self.items = []
        if items is not None:
            self.items = items
        elif split_txt:
            with open(split_txt) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls, img_id = line.split('/')
                    idx = class2idx[cls]
                    path = os.path.join(IM_DIR, cls, img_id + ".jpg")
                    self.items.append((path, idx))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def extract_backend_samples():
    all_images = []
    for cls in classes:
        cls_dir = os.path.join(IM_DIR, cls)
        if os.path.exists(cls_dir):
            images = [os.path.join(cls_dir, img) for img in os.listdir(cls_dir) if img.endswith('.jpg')]
            all_images.extend([(img, cls) for img in images])
    
    if len(all_images) > BACKEND_SAMPLES:
        backend_samples = random.sample(all_images, BACKEND_SAMPLES)
        for img_path, cls in backend_samples:
            cls_dir = os.path.join(BACKEND_DIR, cls)
            os.makedirs(cls_dir, exist_ok=True)
            shutil.copy(img_path, os.path.join(cls_dir, os.path.basename(img_path)))
        return [img_path for img_path, _ in backend_samples]
    return []

def get_datasets_and_loaders():
    backend_image_paths = extract_backend_samples()
    backend_image_paths_set = set(backend_image_paths)

    train_items, test_items_full = [], []
    with open(TRAIN_FILE) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            cls, img_id = line.split('/')
            path = os.path.join(IM_DIR, cls, img_id + ".jpg")
            if path in backend_image_paths_set:
                continue
            train_items.append((path, class2idx[cls]))

    with open(TEST_FILE) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            cls, img_id = line.split('/')
            path = os.path.join(IM_DIR, cls, img_id + ".jpg")
            if path in backend_image_paths_set:
                continue
            test_items_full.append((path, class2idx[cls]))

    val_size = int(len(test_items_full) * VAL_SPLIT)
    test_size = len(test_items_full) - val_size
    val_items, test_items = random_split(test_items_full, [val_size, test_size])

    train_tfms, val_tfms = get_transforms()
    train_ds = Food101Dataset(items=train_items, transform=train_tfms)
    val_ds = Food101Dataset(items=val_items, transform=val_tfms)
    test_ds = Food101Dataset(items=test_items, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, classes
