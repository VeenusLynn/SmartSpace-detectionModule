from flask import Flask, request, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the YOLOv8 model
model = YOLO("SmartSpace.pt")

# Class indices
classes_to_detect = [0, 2, 7]


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Perform detection with class limitation
    results = model.predict(img, classes=classes_to_detect, conf=0.5)
    boxes = results[0].boxes.xywh
    classes = results[0].boxes.cls
    names = results[0].names
    confidences = results[0].boxes.conf

    detections = []
    for box, cls, conf in zip(boxes, classes, confidences):
        x, y, w, h = box
        name = names[int(cls)]
        detections.append(
            {
                "class": name,
                "confidence": float(conf),
                "box": [int(x), int(y), int(w), int(h)],
            }
        )
    annotated_img = results[0].plot()

    output_path = "static/annotated.jpg"
    cv2.imwrite(output_path, annotated_img)

    return jsonify({"detections": detections})


@app.route("/annotated-image", methods=["GET"])
def annotated_image():
    return app.send_static_file("annotated.jpg")


if __name__ == "__main__":
    # Run the Flask app on the url http://localhost:5001/
    app.run(host="0.0.0.0", port=5001)
