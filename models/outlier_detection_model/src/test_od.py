# %%
import os
import yaml
import matplotlib.pyplot as plt
import torch
# OD stuff
from alibi_detect.saving import load_detector
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
# Model definition stuff
import tensorflow as tf
# from tensorflow.keras.layers import InputLayer, Conv2D, Conv2DTranspose, Dense, Reshape
from prepare import get_datasets_and_loaders, get_outlier_loader
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
load_dotenv()
# wandb_key = os.getenv("WANDB_API_KEY")
# import wandb
# wandb.login(wandb_key)
current_time = datetime.now()
tstamp = current_time.strftime("%Y-%m-%d_%H:%M:%S")

#Ignore TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# print("Using config:", config)


PIN_MEMORY = False # Improves GPU perf. by enabling faster CPU->GPU loads
NUM_WORKERS = 0

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

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

 # Hyperparameters
BATCH_SIZE = config["train"]["batch_size"]
SAVE_DIR = "models/outlier_detection_model/checkpoints"

train_loader, val_loader, test_loader, classes, train_np = get_datasets_and_loaders(BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
outlier_np, mixed_loader, mixed_np = get_outlier_loader(BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)

outlier_detector = load_detector(SAVE_DIR)


# Convert np array to tf dataset

train_np_tf = np.transpose(train_np, (0, 2, 3, 1))  # To (B, H, W, C)
train_np_tf = tf.image.resize(train_np_tf, (64, 64)).numpy()
print(f"train_np shape: {train_np.shape}, dtype: {train_np.dtype}, size: {train_np.nbytes / 1024**2:.2f} MB")

idx = 1
X = train_np_tf[idx].reshape(1, 64, 64, 3)
# X = X / 255.0  # Normalize to [0, 1] range
print("Train min/max:", train_np_tf.min(), train_np_tf.max())
X_reconstructed = outlier_detector.ae(X)
print("Reconstruction min/max:", tf.reduce_min(X_reconstructed).numpy(), tf.reduce_max(X_reconstructed).numpy())

print("Plotting the original image...")
plt.imshow(X.reshape(64, 64, 3))
plt.axis('off')
plt.show()

print("Plot the reconstructed image...")
plt.imshow(X_reconstructed.numpy().reshape(64, 64, 3))
plt.axis('off')
plt.show()

# Infer threshold

mixed_np_tf = np.transpose(mixed_np, (0, 2, 3, 1))  # To (B, H, W, C)
mixed_np_tf = tf.image.resize(mixed_np_tf, (64, 64)).numpy()
print(f"train_np shape: {mixed_np.shape}, dtype: {mixed_np.dtype}, size: {mixed_np.nbytes / 1024**2:.2f} MB")

print("Inference threshold...")
outlier_detector.infer_threshold(
    mixed_np_tf,
    threshold_perc=83, #percentage of normal data to use for threshold calculation
)

print("Plotting instance-level detected outlier scores on the normal dataset...")
X = train_np_tf[:500]
print(f"Train set shape: {X.shape}")
od_preds = outlier_detector.predict(X,
                      outlier_type='instance',    # use 'feature' or 'instance' level
                      return_feature_score=True,  # scores used to determine outliers
                      return_instance_score=True)
print(f"OD Output Dict Keys: {list(od_preds['data'].keys())}")

target = np.zeros(X.shape[0],).astype(int)  # all normal CIFAR10 training instances
labels = ['normal', 'outlier']
plot_instance_score(od_preds, target, labels, outlier_detector.threshold)

print("Plotting Outlier Score Channels for non-outliers")

X_recon = outlier_detector.ae(X).numpy()
plot_feature_outlier_image(od_preds,
                           X,
                           X_recon=X_recon,
                           instance_ids=[8, 60, 100, 200],  # pass a list with indices of instances to display
                           max_instances=5,  # max nb of instances to display
                           outliers_only=False)  # only show outlier predictions

print("****************************")
print("\n\nRepeat visualisation for outliers")

outlier_np_tf = np.transpose(outlier_np, (0, 2, 3, 1))  # To (B, H, W, C)
outlier_np_tf = tf.image.resize(outlier_np_tf, (64, 64)).numpy()

idx = 1
O = outlier_np_tf[idx].reshape(1, 64, 64, 3)
# X = X / 255.0  # Normalize to [0, 1] range
print("Train min/max:", train_np_tf.min(), train_np_tf.max())
O_reconstructed = outlier_detector.ae(O)
print("Reconstruction min/max:", tf.reduce_min(O_reconstructed).numpy(), tf.reduce_max(O_reconstructed).numpy())

print("Plotting the original outlier...")
plt.imshow(O.reshape(64, 64, 3))
plt.axis('off')
plt.show()

print("Plot the reconstructed outlier...")
plt.imshow(O_reconstructed.numpy().reshape(64, 64, 3))
plt.axis('off')
plt.show()

print("Plotting instance-level detected outlier scores on the abnormal dataset...")
O = outlier_np_tf[:500]
od_preds = outlier_detector.predict(O,
                      outlier_type='instance',    # use 'feature' or 'instance' level
                      return_feature_score=True,  # scores used to determine outliers
                      return_instance_score=True)

target = np.zeros(O.shape[0],).astype(int)  # all normal CIFAR10 training instances
labels = ['abnormal', 'outlier']
plot_instance_score(od_preds, target, labels, outlier_detector.threshold)

print("Plotting Outlier Score Channels for non-outliers")

O_recon = outlier_detector.ae(O).numpy()
plot_feature_outlier_image(od_preds,
                           O,
                           X_recon=O_recon,
                           instance_ids=[8, 60, 65, 70],  # pass a list with indices of instances to display
                           max_instances=5,  # max nb of instances to display
                           outliers_only=False)  # only show outlier predictions


#%%