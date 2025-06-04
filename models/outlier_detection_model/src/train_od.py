#%% 

import os
import yaml
import torch
# OD stuff
import alibi_detect
from alibi_detect.od import OutlierAE
from alibi_detect.saving import save_detector
# Model definition stuff
import tensorflow as tf
from keras.layers import InputLayer, Conv2D, Conv2DTranspose, Dense, Reshape, Flatten
# from tensorflow.keras.layers import InputLayer, Conv2D, Conv2DTranspose, Dense, Reshape
from prepare import get_datasets_and_loaders, get_outlier_loader
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import sys

# Open a log file in write mode
log_file = open('./AE_log_200.txt', 'w')

# Redirect stdout to the log file
sys.stdout = log_file

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
NUM_EPOCHS = config["train"]["num_epochs"]
LR = config["train"]["learning_rate"]
WEIGHT_DECAY = config["train"]["weight_decay"]
PATIENCE = config["train"]["patience"]
HIDDEN_DIM = config["train"]["hidden_dim"]
SAVE_DIR = "".join(PROJECT_DIR.split("/")[:-2]) + "/models/outlier_detection_model/checkpoints" #TODO - set this to your own dir

train_loader, val_loader, test_loader, classes, train_np = get_datasets_and_loaders(BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
outlier_loader, mixed_loader, mixed_np = get_outlier_loader(BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)


print("Defining model...")

encoder = tf.keras.Sequential(
  [
      InputLayer(input_shape=(64, 64, 3)),
      Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(256, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
      Flatten(),
      Dense(512, )
  ])

decoder = tf.keras.Sequential(
  [
      InputLayer(input_shape=(512,)),
      Dense(4*4*512),
      Reshape(target_shape=(4, 4, 512)),
      Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu'),
      Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
  ])


print("Defining outlier detector...")

outlier_detector = OutlierAE(
    threshold=0.1,
    encoder_net=encoder,
    decoder_net=decoder
)

# Convert np array to tf dataset

train_np_tf = np.transpose(train_np, (0, 2, 3, 1))  # To (B, H, W, C)
train_np_tf = tf.image.resize(train_np_tf, (64, 64)).numpy()


print(f"train_np shape: {train_np.shape}, dtype: {train_np.dtype}, size: {train_np.nbytes / 1024**2:.2f} MB")

outlier_detector.fit(
    train_np_tf,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=True,
)

print("Training complete.")

mixed_np_tf = np.transpose(mixed_np, (0, 2, 3, 1))  # To (B, H, W, C)
mixed_np_tf = tf.image.resize(mixed_np_tf, (64, 64)).numpy()
print(f"train_np shape: {mixed_np.shape}, dtype: {mixed_np.dtype}, size: {mixed_np.nbytes / 1024**2:.2f} MB")

print("Inference threshold...")
outlier_detector.infer_threshold(
    mixed_np_tf,
    threshold_perc=83, #percentage of normal data to use for threshold calculation
)

save_detector(outlier_detector, SAVE_DIR)
print("Detector saved to:", SAVE_DIR)

log_file.close()
# %%
