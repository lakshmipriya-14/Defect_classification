import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tensorflow as tf

# Initialize Flask App
app = Flask(__name__)
app.secret_key = "steel_defect_classification_secret_key"

# Configuration
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload directory if it does not exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Image parameters extracted strictly from Notebook
IMG_SIZE = (200, 200)

# Class mappings based on Keras flow_from_directory alphabetical folder order.
# Notebook standard folder names: ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
# Map index -> (Display Name, Description)
CLASS_MAPPING = {
    0: ("Crazing", "Fine network of micro-cracks on the steel surface caused by thermal or mechanical stress."),
    1: ("Inclusion", "Non-metallic particles or impurities entrapped in the steel matrix during casting."),
    2: ("Pitted Surface", "Small cavity depressions or pits formed due to corrosion or chemical reaction."),
    3: ("Patches", "Irregular surface discoloration or localized structural defects on the sheet."),
    4: ("Rolled-in Scale", "Oxide scale pressed into the metal surface during hot rolling operations."),
    5: ("Scratches", "Linear gouges or abrasions caused by mechanical friction during transport or handling.")
}

# Alternative mapping if dataset subfolders were renamed to short codes ["Cr", "In", "PS", "Pa", "RS", "Sc"]
# Note: ASCII order of short codes places "PS" before "Pa"
SHORT_CODE_MAPPING = {
    0: ("Crazing (Cr)", "Micro-cracks on the steel surface."),
    1: ("Inclusion (In)", "Non-metallic impurities entrapped in steel."),
    2: ("Pitted Surface (PS)", "Small cavity depressions or pits."),
    3: ("Patches (Pa)", "Irregular surface discoloration or defects."),
    4: ("Rolled-in Scale (RS)", "Oxide scale pressed into the metal surface."),
    5: ("Scratches (Sc)", "Linear gouges or surface abrasions.")
}

# Model path resolution (Check for trained_model.keras or product_defect_cnn_model.h5)
MODEL_PATHS = [
    os.path.join("model", "trained_model.keras"),
    os.path.join("model", "product_defect_cnn_model.h5"),
    "trained_model.keras",
    "product_defect_cnn_model.h5"
]

model = None
loaded_model_path = None

def load_defect_model():
    """Loads the trained Keras model from disk."""
    global model, loaded_model_path
    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                loaded_model_path = path
                print(f"[INFO] Model loaded successfully from '{path}'.")
                return model
            except Exception as e:
                print(f"[ERROR] Failed to load model from '{path}': {e}")
    print("[WARNING] No trained model found in 'model/' directory. Please place 'trained_model.keras' or 'product_defect_cnn_model.h5' inside 'model/'.")
    return None

# Load model on startup
load_defect_model()

def allowed_file(filename):
    """Checks if uploaded file extension is supported."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """
    Preprocesses the uploaded image to match notebook training pipeline:
    1. Load image and convert to RGB (3 channels)
    2. Resize to (200, 200)
    3. Scale pixel values by 1/255.0
    4. Expand dimensions to (1, 200, 200, 3)
    """
    img = Image.open(image_path)
    
    # Ensure 3-channel RGB (handles grayscale or RGBA PNGs)
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    # Resize image to target size matching notebook target_size=(200, 200)
    img = img.resize(IMG_SIZE)
    
    # Convert image to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Rescale pixel values (matching ImageDataGenerator(rescale=1./255))
    img_array = img_array / 255.0
    
    # Expand dimensions for batch input (1, 200, 200, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route("/", methods=["GET"])
def index():
    """Renders the main upload page."""
    model_status = loaded_model_path is not None or model is not None
    return render_template("index.html", model_loaded=model_status)

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload and prediction."""
    global model
    
    # 1. Verify model is loaded
    if model is None:
        model = load_defect_model()
        if model is None:
            flash("Model file not found! Please place 'trained_model.keras' or 'product_defect_cnn_model.h5' inside the 'model/' directory.", "error")
            return redirect(url_for("index"))

    # 2. Check if file is part of request
    if "file" not in request.files:
        flash("No file part in the request.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]

    # 3. Check if user submitted empty file selection
    if file.filename == "":
        flash("No image selected for upload. Please select a steel surface image.", "error")
        return redirect(url_for("index"))

    # 4. Validate file type
    if not allowed_file(file.filename):
        flash("Unsupported file format! Allowed formats are: JPG, JPEG, PNG, BMP.", "error")
        return redirect(url_for("index"))

    try:
        # 5. Save file securely to upload directory
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # 6. Preprocess image strictly matching training pipeline
        img_tensor = preprocess_image(filepath)

        # 7. Model inference
        predictions = model.predict(img_tensor)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx]) * 100.0

        # 8. Retrieve class info
        class_name, description = CLASS_MAPPING.get(
            predicted_idx, ("Unknown Defect", "No description available for this class.")
        )

        # Build list of class probabilities for transparent breakdown
        class_probabilities = []
        for idx, prob in enumerate(predictions[0]):
            name, _ = CLASS_MAPPING.get(idx, (f"Class {idx}", ""))
            class_probabilities.append({
                "class_name": name,
                "probability": round(float(prob) * 100, 2)
            })

        # Relative web path for display in template
        image_url = url_for("static", filename=f"uploads/{filename}")

        return render_template(
            "result.html",
            predicted_class=class_name,
            confidence=f"{confidence:.2f}",
            description=description,
            image_url=image_url,
            filename=filename,
            probabilities=class_probabilities
        )

    except Exception as e:
        flash(f"An error occurred during prediction: {str(e)}", "error")
        return redirect(url_for("index"))

if __name__ == "__main__":
    print("[INFO] Starting Steel Defect Classification Web Application...")
    print("[INFO] Access the web UI at: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
