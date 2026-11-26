
from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import numpy as np
import os
import json

app = Flask(__name__)

# =====================================
# PROJECT SETTINGS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_efficientnetb0_3class.keras"
)

MAPPING_PATH = os.path.join(
    BASE_DIR,
    "class_mapping_3class.json"
)

IMG_SIZE = 224

CLASS_LABELS = {
    "real_notes": "Genuine Banknote",
    "fake_notes": "Fake Banknote",
    "not_banknotes": "Not a Banknote"
}

# =====================================
# LOAD CLASS MAPPING
# =====================================

print("Loading class mapping...")

with open(MAPPING_PATH, "r") as file:
    class_mapping = json.load(file)

INDEX_TO_CLASS = {
    index: name
    for name, index in class_mapping.items()
}

print("Class mapping:", INDEX_TO_CLASS)

# =====================================
# LOAD MODEL
# =====================================

print("\nLoading EfficientNetB0 Model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found: " + MODEL_PATH)

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

if model.output_shape[-1] != 3:
    raise ValueError("Expected a 3-class model.")

# =====================================
# WEBSITE HTML
# =====================================

HTML = """
<!DOCTYPE html>
<html>
<head>

<title>Bangladeshi Banknote Detection</title>

<meta name="viewport" content="width=device-width, initial-scale=1">

<style>

body {
    font-family: Arial, sans-serif;
    background: #eef3f1;
    text-align: center;
    padding: 20px;
}

.container {
    background: white;
    max-width: 520px;
    margin: 25px auto;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px #0002;
}

h1 {
    color: #087f5b;
}

.section {
    background: #f5f8f7;
    padding: 18px;
    border-radius: 10px;
    margin-top: 20px;
}

input {
    margin: 15px;
    max-width: 100%;
}

button {
    background: #087f5b;
    color: white;
    border: none;
    padding: 12px 20px;
    border-radius: 7px;
    cursor: pointer;
    font-size: 15px;
    margin: 5px;
}

button:disabled {
    background: gray;
    cursor: not-allowed;
}

#preview {
    display: none;
    max-width: 100%;
    max-height: 280px;
    margin: 15px auto;
    border-radius: 8px;
}

#cameraVideo {
    display: none;
    width: 100%;
    max-height: 350px;
    background: black;
    border-radius: 10px;
    margin-top: 12px;
}

#cameraCanvas {
    display: none;
}

#result {
    margin-top: 22px;
    font-size: 23px;
    font-weight: bold;
}

#details {
    margin-top: 15px;
    color: #555;
    font-size: 14px;
    line-height: 1.8;
}

.probability {
    background: #f2f7f5;
    padding: 15px;
    border-radius: 8px;
    text-align: left;
}

#cameraMessage {
    font-size: 13px;
    color: #555;
    margin-top: 10px;
}

</style>
</head>

<body>

<div class="container">

    <h1>Bangladeshi Banknote Detection</h1>

    <p>Upload or take a photo of a Bangladeshi banknote.</p>

    <!-- IMAGE UPLOAD SECTION -->

    <div class="section">

        <h3>Upload Image</h3>

        <input
            type="file"
            id="imageInput"
            accept="image/*"
        >

        <br>

        <img id="preview">

    </div>

    <!-- CAMERA SECTION -->

    <div class="section">

        <h3>Open Camera</h3>

        <button id="startCameraButton" onclick="startCamera()">
            Start Camera
        </button>

        <button id="captureButton" onclick="capturePhoto()" disabled>
            Capture Photo
        </button>

        <button id="stopCameraButton" onclick="stopCamera()" disabled>
            Stop Camera
        </button>

        <video
            id="cameraVideo"
            autoplay
            playsinline
            muted
        ></video>

        <canvas id="cameraCanvas"></canvas>

        <div id="cameraMessage">
            Click Start Camera to use your webcam.
        </div>

    </div>

    <!-- PREDICTION BUTTON -->

    <div style="margin-top:20px;">

        <button
            id="checkButton"
            onclick="predict()"
            disabled
        >
            Check Banknote
        </button>

    </div>

    <div id="result"></div>

    <div id="details"></div>

</div>

<script>

// =====================================
// GLOBAL VARIABLES
// =====================================

let selectedFile = null;
let cameraStream = null;

const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");

const video = document.getElementById("cameraVideo");
const canvas = document.getElementById("cameraCanvas");

const checkButton = document.getElementById("checkButton");
const captureButton = document.getElementById("captureButton");

const result = document.getElementById("result");
const details = document.getElementById("details");

// =====================================
// IMAGE UPLOAD
// =====================================

imageInput.addEventListener("change", function() {

    const file = this.files[0];

    if (!file) return;

    selectedFile = file;

    stopCamera();

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";

    checkButton.disabled = false;

    result.innerText = "";
    details.innerHTML = "";

});

// =====================================
// START CAMERA
// =====================================

async function startCamera() {

    try {

        if (!navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia) {

            throw new Error(
                "Camera is not supported in this browser."
            );

        }

        stopCamera();

        cameraStream = await navigator.mediaDevices.getUserMedia({

            video: {
                facingMode: "environment"
            },

            audio: false

        });

        video.srcObject = cameraStream;

        video.style.display = "block";

        await video.play();

        captureButton.disabled = false;

        document.getElementById("stopCameraButton").disabled = false;

        document.getElementById("cameraMessage").innerText =
            "Camera is running. Place the banknote in front of the camera.";

        preview.style.display = "none";

        selectedFile = null;

        imageInput.value = "";

        checkButton.disabled = true;

    }

    catch (error) {

        document.getElementById("cameraMessage").innerText =
            "Camera Error: " + error.message;

    }

}

// =====================================
// CAPTURE PHOTO
// =====================================

function capturePhoto() {

    if (!cameraStream || video.readyState < 2) {

        alert("Camera is not ready.");
        return;

    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );

    canvas.toBlob(function(blob) {

        if (!blob) {

            alert("Could not capture image.");
            return;

        }

        selectedFile = new File(
            [blob],
            "camera_capture.jpg",
            { type: "image/jpeg" }
        );

        preview.src = URL.createObjectURL(selectedFile);

        preview.style.display = "block";

        checkButton.disabled = false;

        result.innerText = "";
        details.innerHTML = "";

        document.getElementById("cameraMessage").innerText =
            "Photo captured successfully.";

    }, "image/jpeg", 0.95);

}

// =====================================
// STOP CAMERA
// =====================================

function stopCamera() {

    if (cameraStream) {

        cameraStream.getTracks().forEach(function(track) {
            track.stop();
        });

        cameraStream = null;

    }

    video.srcObject = null;
    video.style.display = "none";

    captureButton.disabled = true;

    document.getElementById("stopCameraButton").disabled = true;

    document.getElementById("cameraMessage").innerText =
        "Camera stopped.";

}

// =====================================
// PREDICTION
// =====================================

async function predict() {

    if (!selectedFile) {

        result.innerText = "Please upload or capture an image first.";
        return;

    }

    const formData = new FormData();

    formData.append("image", selectedFile);

    result.innerText = "Checking image...";
    details.innerHTML = "";

    checkButton.disabled = true;

    try {

        const response = await fetch("/predict", {

            method: "POST",
            body: formData

        });

        const data = await response.json();

        if (!response.ok) {

            throw new Error(data.error || "Prediction failed");

        }

        result.innerText = data.result;

        details.innerHTML = `

            <div class="probability">

                <strong>Prediction Probabilities</strong>

                <br><br>

                Genuine Banknote:
                ${data.probabilities.real_notes}%

                <br>

                Fake Banknote:
                ${data.probabilities.fake_notes}%

                <br>

                Not a Banknote:
                ${data.probabilities.not_banknotes}%

            </div>

        `;

    }

    catch (error) {

        result.innerText = "Error: " + error.message;

    }

    finally {

        checkButton.disabled = false;

    }

}

</script>

</body>
</html>
"""

# =====================================
# HOME PAGE
# =====================================

@app.route("/")
def home():

    return render_template_string(HTML)


# =====================================
# IMAGE PREPROCESSING
# =====================================

def preprocess_image(image_bytes):

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# =====================================
# PREDICTION ROUTE
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if not file.filename:

        return jsonify({
            "error": "Please select an image"
        }), 400

    try:

        image_bytes = file.read()

        if not image_bytes:

            raise ValueError("Uploaded image is empty.")

        # Verify image
        import io

        with Image.open(io.BytesIO(image_bytes)) as img:

            print("\nOriginal image size:", img.size)
            print("Original image mode:", img.mode)

        # Preprocess
        image_array = preprocess_image(image_bytes)

        print("\n================================")
        print("NEW IMAGE PREDICTION")
        print("Filename:", file.filename)
        print("Processed shape:", image_array.shape)

        print(
            "Pixel minimum:",
            float(tf.reduce_min(image_array))
        )

        print(
            "Pixel maximum:",
            float(tf.reduce_max(image_array))
        )

        # Predict
        prediction = model.predict(
            image_array,
            verbose=0
        )

        print("Raw model output:", prediction)

        if prediction.shape != (1, 3):

            raise ValueError("Unexpected model output shape.")

        probabilities = prediction[0]

        if not np.all(np.isfinite(probabilities)):

            raise ValueError("Invalid prediction.")

        if np.any(probabilities < 0) or np.any(probabilities > 1):

            raise ValueError("Invalid probability range.")

        if not np.isclose(np.sum(probabilities), 1.0, atol=0.01):

            raise ValueError("Invalid probability sum.")

        predicted_index = int(np.argmax(probabilities))

        predicted_class = INDEX_TO_CLASS[predicted_index]

        result = CLASS_LABELS[predicted_class]

        probability_results = {}

        for class_name, class_index in class_mapping.items():

            probability_results[class_name] = round(
                float(probabilities[class_index]) * 100,
                2
            )

        print("\n----- PREDICTION DETAILS -----")
        print("Predicted class:", predicted_class)
        print("Final result:", result)
        print("Probabilities:", probability_results)
        print("------------------------------\n")

        return jsonify({

            "result": result,

            "predicted_class": predicted_class,

            "probabilities": probability_results

        })

    except (UnidentifiedImageError, OSError):

        return jsonify({
            "error": "The uploaded file is not a valid image."
        }), 400

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 400


# =====================================
# RUN SERVER
# =====================================

if __name__ == "__main__":

    app.run(
    host="0.0.0.0",
    port=5000,
    debug=False
)