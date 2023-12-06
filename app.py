import tensorflow as tf
from flask import Flask, jsonify, request
from utils import *
import os

app = Flask(__name__)

# Set the upload directory
UPLOAD_FOLDER = './Upload/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return "App is running"


@app.route("/predict", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file found"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    prediction = predict(file_path,model)
    
    return jsonify(prediction),200


if __name__ == "__main__":
    #Loading the Model
    model = tf.keras.models.load_model('./Model/model_2_pohon.h5')
    app.run(debug=True)
