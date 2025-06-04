# 🍽️ NexDish Setup Instructions

Follow the steps below to set up and run the NexDish project locally.

---

## 🔧 1. Clone the Repository

```bash
git clone https://github.com/MelvinPariyadan/NexDish.git
cd NexDish
```

---

## 🐍 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate  # On macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

---

## 🔐 3. Set Up DVC for Dataset & Model Syncing

> ⚠️ The dataset and model files are stored in an AWS S3 bucket and are managed using DVC.
To access them, you’ll need two secret key files, which are available in the **Project Secrets** folder on OneDrive for this project.

1. Add your provided **`.env`** file in the project root. So it will be in NexDish/.env
2. Add the provided **`config`** file in the `.dvc/` folder. So it will be in NexDish/.dvc/config. It contains the correct S3 bucket configurations.
3. Pull the dataset and model (this can take up to **30 minutes** depending on your network):

```bash
dvc pull
```

---

## 🐳 4. Run with Docker

Make sure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running.

```bash
docker-compose up -d
# If you make a change to code for example frontend and want to see it after running this command first run
# docker-compose down --rmi all --volume, to remove existing container and then run docker-compose up -d again it will have your code changes 
```

---

## 🌐 5. Access the App

Open your browser and go to:

```
http://localhost:8501
```




# 🧠 Training Instructions

All code related to model training is located in:

```
models/food_classification_model/src/
```

To start training, run:

```bash
python models/food_classification_model/src/train.py
```

### 🔧 Code Structure

- `params.yaml` – Centralized configuration for hyperparameters and paths  
- `model.py` – ResNet-50 model architecture definition  
- `transforms.py` – Image preprocessing and data augmentation  
- `prepare.py` – Dataset loading and splitting into train/val/test  
- `train.py` – Main training loop and saving model

The trained model is saved at the path specified in `params.yaml`.

### 📊 Accuracy Testing

After training the model, we manually tested its accuracy using the .pth file saved in the checkpoints folder.

To run the manual accuracy test:

```bash
python tests/test_model_accuracy.py
```
This script loads the trained model and checks if its accuracy meets the expected threshold (e.g., ≥ 85%).

## Working with DVC

We use **DVC (Data Version Control)** to track and version the dataset and model weights. Anytime you update the dataset or model weights and we wanna track those with git, we use the following commands:

### Step 1: Add Changes to DVC

Use the `dvc add` command to track changes:

```bash
dvc add dataset  # Run this if you modified the dataset
dvc add models/food_classification_model/checkpoints  # Run this if new model weights were added
dvc add models/outlier_detection_model/checkpoints # Outlier Detection models
```

This updates the corresponding `.dvc` files (`dataset.dvc`, `checkpoints.dvc`).

### Step 2: Commit Changes to Git

After updating the DVC files, commit them along with any other changes (e.g., updated `params.yaml`):

```bash
git add dataset.dvc models/food_classification_model/checkpoints.dvc params.yaml
git commit -m "Updated dataset and model checkpoints"
git push
```

### Step 3: Push to Remote Storage

Finally, push the DVC-tracked data to the remote storage (e.g., S3 bucket):

```bash
dvc push
```

This ensures your data and model versions are stored and shared properly across collaborators or deployments.

