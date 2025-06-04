from torchvision import transforms

def get_transforms():
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

        # Strong, float32-compatible distortions
        transforms.RandomRotation(degrees=90),                    # Huge rotation
        transforms.RandomHorizontalFlip(p=1.0),                   # Always flip
        transforms.ColorJitter(brightness=2, contrast=2, 
                            saturation=2, hue=0.5),            # Aggressive color changes
        transforms.GaussianBlur(kernel_size=9, sigma=(5.0, 10.0)), # Strong blur
        transforms.RandomErasing(p=1.0, scale=(0.1, 0.4)),        # Cutout-like region masking
    ])

    return train_tfms, val_tfms
