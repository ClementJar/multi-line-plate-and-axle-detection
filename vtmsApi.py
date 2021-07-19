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
    detect_plate = "python detect_plate.py "
    detect_axle = "python axle_count.py "
    plate_weight = " --weights ./checkpoints/yolov4-tiny-license-plate-416 --size 416 "
    axle_weight = " --weights ./checkpoints/axle-yolov4-final-416 --size 416 "
    tiny_weight_and_size = " --weights ./checkpoints/yolov4-License-Plate-416 --size 416 "
    model = " --model yolov4 "
    video_par = " --video "
    ip_address_par = " --ip_address "
    vehicle_side_par = " --vehicle_side "  # back or front of vehicle
    gate_id_par = " --gate_id "  # back or front of vehicle
    username_par = " --username "
    password_par = " --password "
    front_right_par = " --front_right "
    front_left_par = " --front_left "
    back_right_par = " --back_right "
    back_left_par = " --back_left "
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
        det_type = data['type']  # type detection to be performed, image or vide
        password = data['password']  # password of camera
        user_name = data['userName']  # username of camera
        int_port = data['intPort']  # internal port of camera
        ip_address = data['ipAddress']  # ip address of device capturing
        vehicle_side = data['vehicleSide']  # ip address of device capturing
        gate_id = data['gateId']  # ip address of device capturing
        d_count = bool(data['count'])  # whether to count objects or not
        d_crop = bool(data['crop'])  # whether to crop media or not
        d_plate = bool(data['plate'])  # plate recognition or not
        d_tiny = bool(data['tiny'])  # use tiny weights or not
        d_type = bool(data['detectionType'])  # is auto detection or manual

        # 360 image ip adresses
        front_right = data['frontRightIp']  # ip address of device capturing front right o vehicle
        front_left = data['frontLeftIp']  # ip address of device capturing front left of vehicle
        back_right = data['backRightIp']  # ip address of device capturing back right of vehicle
        back_left = data['backLeftIp']  # ip address of device capturing back left of vehicle

        file_name, file_ext = os.path.splitext(path)
        output_file = path + "-" + output_file.strftime("%H:%M:%S") + file_ext
        # camera_url = "rtsp://"+"admin"+ ":" +"Admin1234"+"@"+ ip_address + ":" +"554"+ "/Streaming/Channels/1"
        camera_url = "rtsp://" + user_name + ":" + password + "@" + ip_address + ":" + int_port + "/Streaming/Channels/1"
        axle_test = "./data/video/axle1.mp4"
        plate_front = "./data/video/plate4.mp4"
        plate_back = "./data/video/plate1.mp4"

        # check if auto
        if d_type:
            command = detect_axle + axle_weight + plate_weight + model + video_par + camera_url
        else:
            command = detect_plate + plate_weight + model + video_par + camera_url
            # check if a video is to be used for detection
        if vehicle_side == "FRONT":
            front_left = ip_address
            front_right = ip_address
            back_right = ip_address
            back_left = ip_address
            # add 360 ip addresses....(i know the command is extra extra long now.. but)
            command = command + back_left_par + back_left + back_right_par + back_right + front_left_par + front_left + front_right_par + front_right

        # check if the tiny weights are to be used
        if d_tiny:
            command = command + tiny_par
        # check if the count method is to be used
        if d_count:
            command = command + count_par
        # # check if the crop method is to be used
        # if d_crop and vehicle_side != "SIDE":
        #     command = command + crop_par
        # # check if the license plate method is to be used
        # if d_plate and vehicle_side != "SIDE":
        #     command = command + plate_par

        # add ip adress to command
        command = command + ip_address_par + ip_address

        # add vehicle direction parameters
        command = command + vehicle_side_par + vehicle_side

        # add gate parameter
        command = command + gate_id_par + gate_id

        return command, ip_address

    except (KeyError, TypeError, ValueError):
        subprocess.Popen(command)

        return


def run_detection(command):
    subprocess.Popen(command)

    print("running in back ground")


@app.route("/run-license-detection", methods=['POST'])
def main():
    data = request.get_json(force=True)
    command, ip_address = prepare_command(data)
    run_detection(command)
    json_response(status=200, message="Detection running on" + ip_address)
    return "Detection running on" + ip_address, 200


## picture detection method

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
