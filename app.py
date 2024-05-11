import io
import json

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, make_response, redirect, render_template

app = Flask(__name__)

import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model_1 = torch.hub.load('./yolov', 'custom', path='./model/number_detect_best.pt', source='local')  # local repo#
model = torch.hub.load('./yolov', 'custom', path='./model/license_plate_best.pt', source='local')  # local repo
model.conf = 0.45
model_1.conf = 0.45


def get_image_from_bytes(binary_image, max_size=1024):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize(
        (
            int(input_image.width * resize_factor),
            int(input_image.height * resize_factor),
        )
    )
    return resized_image


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img, size=640)
        bbox_raw = results.xyxy[0][0]
        bbox = []
        for bound in bbox_raw:
            bbox.append(int(bound.item()))
        bbox = bbox[:4]
        img_array = np.array(img)
        crop_img = img_array[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        res2 = model_1(crop_img, size=640)


        res2.render()  # updates results.imgs with boxes and labels
        for img in res2.ims:
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image0.jpg", format="JPEG")
        return redirect("static/image0.jpg")

        # for debugging
        # data = results.pandas().xyxy[0].to_json(orient="records")
        # return data

    return render_template("index.html")


ALLOWED_EXTENSIONS = ['mp4']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/object-to-json", methods=['GET', 'POST'])
def object_detection():
    if request.method == 'POST':
        if 'video' not in request.files:
            response = jsonify({'error': 'no video file found'})
            return make_response(response, 400)

        video = request.files['video']

        if video.filename == '':
            response = jsonify({'error': 'no video file selected'})
            return make_response(response, 400)

        if video and allowed_file(video.filename):
            results = []
            for frame in video:  # Assuming video is iterable like a file
                input_image = get_image_from_bytes(frame)
                res = model(input_image)
                bbox_raw = res.xyxy[0][0]
                bbox = []
                for bound in bbox_raw:
                    bbox.append(int(bound.item()))
                bbox = bbox[:4]
                crop_img = input_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                res2 = model_1(crop_img)
                res2.pandas().xyxy[0]

            response = jsonify({"results": res2})
            return make_response(response, 200)

    response = jsonify({'error': 'invalid request'})
    return make_response(response, 400)


if __name__ == '__main__':
    app.run()
