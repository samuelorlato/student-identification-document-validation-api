from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import re
from io import BytesIO
import cv2
import numpy as np

app = Flask(__name__)

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def has_one_face(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
    
    for _ in range(3):
        if len(faces) == 0:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
        else:
            break

    return len(faces) == 1

@app.route("/validate_document", methods=["POST"])
def validate_document():
    if "id" not in request.files:
        return jsonify({"error": "Any image found"})

    img = request.files["id"].read()
    img = Image.open(BytesIO(img))

    img_array = np.array(img)

    preprocessed_img = preprocess_image(img_array)

    face = has_one_face(preprocessed_img)
    if face:
        extracted_data = pytesseract.image_to_string(Image.fromarray(preprocessed_img))

        rm_matches = re.findall(r"\b\d{5,}\b", extracted_data)
        if rm_matches:
            rm = rm_matches[0]
            result = {"result": "Valid document", "RM": rm}
            return jsonify(result)
        else:
            return jsonify({"result": "Invalid document"})
    else:
        return jsonify({"result": "Invalid document"})

if __name__ == "__main__":
    app.run(debug=True)
