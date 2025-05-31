# prepare.py
import os
import random
import shutil
import yaml
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from transform import get_transforms
import numpy as np
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_DIR = os.path.dirname(__file__) #make paths absolute
CONFIG_PATH = os.path.join(PROJECT_DIR, 'params.yaml')

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config["train"]["batch_size"]
VAL_SPLIT = config["dataset"]["val_split"]
BACKEND_SAMPLES = config["dataset"]["backend_samples"]

DATA_DIR = 'datasets'
IM_DIR = os.path.join(DATA_DIR, "food-101/images")
OUTLIER_DIR = os.path.join(DATA_DIR, "tau-vehicles/images")

classes = sorted([d for d in os.listdir(IM_DIR) if os.path.isdir(os.path.join(IM_DIR, d))])
class2idx = {c: i for i, c in enumerate(classes)}
idx2class = {i: c for c, i in class2idx.items()}

outlier_classes = sorted([d for d in os.listdir(OUTLIER_DIR) if os.path.isdir(os.path.join(OUTLIER_DIR, d))])
outlier_class2idx = {c: i for i, c in enumerate(outlier_classes)}
outlier_idx2class = {i: c for c, i in outlier_class2idx.items()}

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
    
class OutlierDataset(Dataset):
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

    # Sample a fraction of the train dataset indices
    train_np_fraction = 0.1
    sample_count = int(len(train_ds) * train_np_fraction)
    sampled_indices = random.sample(range(len(train_ds)), sample_count)

    all_train_images = []
    for i in tqdm(sampled_indices, desc=f"Converting {train_np_fraction*100}% train dataset to numpy"):
        img, _ = train_ds[i]
        if hasattr(img, "cpu"):
            img_np = img.cpu().numpy()
        else:
            img_np = np.array(img)
        all_train_images.append(img_np)
    all_train_images = np.stack(all_train_images)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader, test_loader, classes, all_train_images

def get_outlier_loader(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True):
    # Get some valid data to mix with the outlier (for threshold inference)
    train_items = []
    for cls in outlier_classes:
        cls_dir = os.path.join(OUTLIER_DIR, cls)
        images = [img for img in os.listdir(cls_dir) if img.endswith(".jpg")]
        random.shuffle(images)
        train_count = int(0.5 * len(images))  # Take 10% of each class
        cls_train = images[:train_count]

        for img in cls_train:
            path = os.path.join(cls_dir, img)
            train_items.append((path, outlier_class2idx[cls]))

    train_tfms, val_tfms = get_transforms()
    train_ds = Food101Dataset(items=train_items, transform=train_tfms)

    # Outlier data
    outlier_items, test_items_full = [], []
    for cls in outlier_classes:
        cls_dir = os.path.join(OUTLIER_DIR, cls)
        images = [img for img in os.listdir(cls_dir) if img.endswith(".jpg")]
        random.shuffle(images)
        split_idx = int(0.1 * len(images))
        cls_train = images[:split_idx]
        cls_test = images[split_idx:]

        for img in cls_train:
            path = os.path.join(cls_dir, img)
            outlier_items.append((path, outlier_class2idx[cls]))

        for img in cls_test:
            path = os.path.join(cls_dir, img)
            test_items_full.append((path, outlier_class2idx[cls]))

    outlier_ds = OutlierDataset(items=outlier_items, transform=train_tfms)
    true_outlier_ds = OutlierDataset(items=outlier_items, transform=val_tfms)
    print(f"\n\nOutlier dataset size: {len(outlier_ds)}")
    print(f"Train dataset size: {len(train_ds)}")
    mixed_ds = ConcatDataset([train_ds, outlier_ds])
    print("Concat dataset size: ", len(mixed_ds))
    outlier_loader = DataLoader(true_outlier_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    mixed_loader = DataLoader(mixed_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)


    print("Data loaders created")
    X_list = []
    for batch in mixed_loader:
        X, y = batch
        X_list.append(X.numpy())
        # y_list.append(y.numpy())
    mixed_x_np = np.concatenate(X_list, axis=0)
    # repeat for raw outliers
    X_list = []
    for batch in outlier_loader:
        X, y = batch
        X_list.append(X.numpy())
        # y_list.append(y.numpy())
    outlier_np = np.concatenate(X_list, axis=0)
    print("Numpy arrays created")


    return outlier_np, mixed_loader, mixed_x_np