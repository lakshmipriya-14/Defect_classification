# Steel Surface Defect Classification Web Application

A Flask web application for deploying a deep-learning Convolutional Neural Network (CNN) model trained to classify surface defects on hot-rolled steel strips using the **NEU Surface Defect Database**.

---

## 📌 Project Overview

This application deploys a custom 3-block CNN built with TensorFlow/Keras. It allows manufacturing operators and engineers to upload steel surface images and immediately receive:
- **Predicted Defect Class** (Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches)
- **Prediction Confidence Percentage** (e.g., `94.32%`)
- **Visual Probability Breakdown** across all 6 defect categories
- **Defect Description & Characteristics**

---

## 🛠️ System Requirements

- **Operating System**: Windows 10 / 11
- **IDE**: Visual Studio Code (VS Code)
- **Python**: Version 3.8 to 3.11
- **Terminal**: Windows PowerShell or Command Prompt

---

## 📂 Project Structure

```text
steel_defect_classification/
│
├── app.py                      # Flask Application Backend & Model Inference
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Documentation
│
├── model/
│   ├── README.md               # Model placement guide
│   └── trained_model.keras     # Place your trained Keras model here
│
├── static/
│   ├── css/
│   │   └── style.css           # Vanilla CSS Design System
│   └── uploads/                # Temporary storage for uploaded user images
│
└── templates/
    ├── index.html              # Main Upload Interface
    └── result.html             # Classification Result Page
```

---

## 🚀 Setup & Installation Guide (Windows + VS Code)

### Step 1: Open Project in VS Code
1. Open **VS Code**.
2. Go to **File > Open Folder...** and select the `steel_defect_classification` folder.
3. Open a new terminal in VS Code (`Ctrl + ~` or **Terminal > New Terminal**).

### Step 2: Create a Virtual Environment
Run the following command in PowerShell:
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
In PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
*Note: If PowerShell displays an ExecutionPolicy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or activate via CMD using `.\venv\Scripts\activate.bat`.*

### Step 4: Upgrade pip & Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🤖 Model File Placement

1. Train your model in your Jupyter Notebook (`.ipynb`).
2. Save the trained model using:
   ```python
   model.save("model/trained_model.keras")
   ```
   *(or `model.save("product_defect_cnn_model.h5")`)*
3. Copy the saved `.keras` or `.h5` file into the `model/` folder:
   ```text
   steel_defect_classification/model/trained_model.keras
   ```

---

## 🏃 Running the Flask Application

Run the app using Python:
```powershell
python app.py
```

You will see output similar to:
```text
[INFO] Model loaded successfully from 'model/trained_model.keras'.
[INFO] Starting Steel Defect Classification Web Application...
[INFO] Access the web UI at: http://127.0.0.1:5000
 * Running on http://127.0.0.1:5000
```

---

## 🌐 How to Use the Application

1. Open your browser and navigate to: **`http://127.0.0.1:5000`**
2. Click **"Browse Image"** or drag and drop a steel surface image (`.jpg`, `.jpeg`, `.png`, `.bmp`).
3. Click **"Classify Defect"**.
4. View the predicted defect, confidence score, image preview, and detailed category probabilities on the result page.
5. Click **"Classify Another Image"** to analyze a new image.

---

## 🔍 Preprocessing Compatibility Matrix

| Parameter | Notebook Setting | Flask Setting | Verified |
| :--- | :--- | :--- | :--- |
| **Input Shape** | `(200, 200, 3)` | `(200, 200, 3)` | ✅ Exact Match |
| **Rescaling** | `1.0 / 255.0` | `img_array / 255.0` | ✅ Exact Match |
| **Color Mode** | RGB | RGB (`img.convert("RGB")`) | ✅ Exact Match |
| **Classes** | 6 Categorical Classes | 6 Categorical Classes | ✅ Exact Match |
