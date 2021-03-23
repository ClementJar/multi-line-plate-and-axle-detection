import os
import http.client
import json
import subprocess
from threading import Thread
from datetime import datetime

from flask import Flask, request
from flask_json import FlaskJSON, JsonError, json_response, as_json
# comment out below line to enable tensorflow outputs
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = Flask(__name__)


def prepare_command(data):
    detect_with_video = "python detect_video.py "
    detect_with_image = "python detect.py "
    normal_weight_and_size = " --weights ./checkpoints/yolov4-License-Plate-416 --size 416 "
    tiny_weight_and_size = " --weights ./checkpoints/yolov4-License-Plate-416 --size 416 "
    model = " --model yolov4 "
    video_par = " --video "
    image_par = " --images "
    output_par = " --output "
    count_par = " --count "
    crop_par = " --crop "
    plate_par = " --plate "
    tiny_par = " --tiny "
    output_path = " ./detections/"
    output_file = datetime.now()
    # using force inorder to skip mimetype checking to have shorter curl command

    # disable json add status
    app.config['JSON_ADD_STATUS'] = False

    try:

        # starting with a single path, will make it more scalable later
        path = data['path']  # path to the media to be used for detection
        model_path = data['model_path']  # path to the media to be used for detection
        det_type = data['type']  # type detection to be performed, image or vide
        d_count = bool(data['count'])  # whether to count objects or not
        d_crop = bool(data['crop'])  # whether to crop media or not
        d_plate = bool(data['plate'])  # plate recognition or not
        d_tiny = bool(data['tiny'])  # use tiny weights or not

        file_name, file_ext = os.path.splitext(path)
        output_file = path + "-" + output_file.strftime("%H:%M:%S") + file_ext

        # check if the tiny weights are to be used
        if d_tiny:
            d_weight = tiny_weight_and_size
        else:
            d_weight = normal_weight_and_size
        # check if a video is to be used for detection
        if det_type == "video":
            command = detect_with_video + d_weight + model + video_par + path + output_par + output_path + output_file
        else:
            command = detect_with_image + d_weight + model + image_par + path + output_par + output_path + output_file
        # check if the tiny weights are to be used
        if d_tiny:
            command = command + tiny_par
        # check if the count method is to be used
        if d_count:
            command = command + count_par
        # check if the crop method is to be used
        if d_crop:
            command = command + crop_par
        # check if the license plate method is to be used
        if d_plate:
            command = command + plate_par

        return command, path

    except (KeyError, TypeError, ValueError):

        return


def run_detection(command):
    subprocess.Popen(command)

    print("running in back ground")


@app.route("/run-license-detection", methods=['POST'])
def main():
    data = request.get_json(force=True)
    command, path = prepare_command(data)
    run_detection(command)
    json_response(status=200, message="Detection running on" + path)

## picture detection method

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
