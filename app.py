
from flask import Flask, render_template, request
from flask import Flask, render_template, request, redirect, url_for
import os
import argparse
import onnxruntime as ort
import torch
import cv2
from ultralytics.utils import ROOT, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml
from models.yolov8_onnx import Yolov8
# create an app object using Flask
app = Flask(__name__,static_folder='static')

# home route to reder index.html from templates folder
@app.route('/')
def home():
    return render_template('index.html', prediction_text='hello world')


# Homepage
@app.route("/", methods=["GET", "POST"])
def homepage():
    if request.method == "POST":
        if "image" not in request.files:
            print('no image')
            return redirect(request.url)
        image = request.files["image"]
        if image.filename == "":
            return redirect(request.url)
        if image:
            image_path = os.path.join("static/uploads", image.filename)
            image.save(image_path)
            return redirect(url_for("detection", image_filename=image.filename))
    return render_template("index.html")

# Object detection result page
@app.route("/detection/<image_filename>")
def detection(image_filename):
    image_path = os.path.join("static/uploads", image_filename)
    save_image_path = os.path.join("static/detections", image_filename)

    output_image = yolov8.main(image_path)
    cv2.imwrite(save_image_path,output_image)
    return render_template("detection.html", image_filename=image_filename)

if __name__ == "__main__":
        # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='models/yolov8s.onnx', help='Input your ONNX model.')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    # parse port argument if given
    parser.add_argument('--port', type=int, default=5000, help='Port to run the web app on')

    args = parser.parse_args()

    # Check the requirements and select the appropriate backend (CPU or GPU)
    check_requirements('onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime')

    # Create an instance of the Yolov8 class with the specified arguments
    yolov8 = Yolov8(args.model, args.conf_thres, args.iou_thres)
    app.run(debug=True, port=3000,)