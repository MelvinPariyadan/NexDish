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

> ⚠️ The dataset and model files are stored in an AWS S3 bucket and tracked using DVC. Here you need 2 configuration files, which will be provided by Melvin. 

1. Add your provided **`.env`** file in the project root. 
2. Add the provided **`.dvc/config`** file in the `.dvc/` folder. It contains the correct S3 bucket configurations.
3. Pull the dataset and model (this can take up to **30 minutes** depending on your network):

```bash
dvc pull
```

---

## 🐳 4. Run with Docker

Make sure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running.

```bash
docker-compose up -d
```

---

## 🌐 5. Access the App

Open your browser and go to:

```
http://localhost:8501
```

