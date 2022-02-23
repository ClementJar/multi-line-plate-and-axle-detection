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
import psutil

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = Flask(__name__)
path = os.path.dirname(os.path.abspath(__file__))


def prepare_command(data, is_not_plate, only_capture=False):
    detect_plate = "python " + path + "/detect_plate.py "
    detect = "python " + path + "/detect.py "
    capture = "python " + path + "/capture_images.py "
    # detect_axle = "python axle_count.py "
    plate_weight = " --plate_weights " + path + "/checkpoints/yolov4-license-plate-416-2 --size 416 "
    axle_weight = " --axle_weights " + path + "/checkpoints/yolov4-tiny-axle-416 --size 416 "
    tiny_weight_and_size = " --weights " + path + "/checkpoints/yolov4-License-Plate-416 --size 416 "
    model = " --model yolov4 "
    video_par = " --video "
    axle_ip_par = " --axle_ip_address "
    vehicle_side_par = " --vehicle_side "  # back or front of vehicle
    gate_id_par = " --gate_id "  # back or front of vehicle
    username_par = " --username "
    password_par = " --password "
    front_ip_par = " --front_ip_address "
    back_ip_par = " --back_ip_address "
    front_right_par = " --front_right "
    front_left_par = " --front_left "
    back_right_par = " --back_right "
    back_left_par = " --back_left "
    is_plate_par = " --is_plate "
    crop_par = " --crop "
    transaction_par = " --transaction "
    plate_par = " --plate "
    tiny_par = " --tiny "
    tiny_par = " --port "
    output_path = " ./detections/"
    output_file = datetime.now()
    # using force inorder to skip mimetype checking to have shorter curl command

    # disable json add status
    app.config['JSON_ADD_STATUS'] = False

    try:

        # starting with a single path, will make it more scalable later
        det_type = data['type']  # type detection to be performed, image or vide
        password = data['password']  # password of camera
        front_password = data['frontPass']  # password of camera
        back_password = data['backPass']  # password of camera
        user_name = data['userName']  # username of camera
        front_user_name = data['frontUsername']  # username of camera
        back_user_name = data['backUsername']  # username of camera
        int_port = data['intPort']  # internal port of camera
        axle_ip_address = data['axleIp']  # path to the media to be used for detection
        front_ip_address = data['frontIp']  # ip address of device capturing
        back_ip_address = data['backIp']  # ip address of device capturing
        gate_id = data['gateId']  # ip address of device capturing
        transaction_ref = data['transactionRef']  # ip address of device capturing
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

        # camera_url = "rtsp://"+"admin"+ ":" +"Admin1234"+"@"+ ip_address + ":" +"554"+ "/Streaming/Channels/1"
        # camera_url = "rtsp://" + user_name + ":" + password + "@" + axle_ip_address + ":" + int_port + "/Streaming/Channels/1"
        # camera_url = "rtsp://" + "admin" + ":" + "Admin1234" + "@" + axle_ip_address + ":" + int_port + "/Streaming/Channels/101"
        camera_url = "rtsp://" + "admin" + ":" + "Admin1234" + "@" + axle_ip_address + ":" + int_port + "/Streaming/Channels/101"
        axle_test = path + "/data/video/axle1.mp4"
        plate_front = "./data/video/plate4.mp4"
        plate_back = "./data/video/plate1.mp4"
        front_left = front_ip_address
        front_right = front_ip_address
        back_right = front_ip_address
        back_left = front_ip_address
        # type of detection
        if only_capture:
            command = capture
        else:
            command = detect + plate_weight + model
        # license plate method is to be used
        command = command + front_ip_par + front_ip_address + back_ip_par + back_ip_address + transaction_par + transaction_ref
        # add 360 ip addresses....(i know the command is extra extra long now.. but)
        command = command + back_left_par + back_left + back_right_par + back_right + front_left_par + front_left + front_right_par + front_right

        # axle counting data
        command = command + axle_weight + video_par + camera_url
        # add ip adress to command
        command = command + axle_ip_par + axle_ip_address
        # check if auto

        # check if the tiny weights are to be used
        if d_tiny:
            command = command + tiny_par
        # add gate parameter
        command = command + gate_id_par + gate_id

        if is_not_plate:
            return command, axle_ip_address, front_ip_address, back_ip_address + tiny_par
        else:
            command = command + is_plate_par + str(is_not_plate)
            return command, axle_ip_address, front_ip_address, back_ip_address

    except (KeyError, TypeError, ValueError):
        # subprocess.Popen(command)

        return


def run_detection(command):
    subprocess.Popen(command)
    print("running in back ground")


def verification(axle_ip, front_ip, back_ip):
    for pid in psutil.pids():
        p = psutil.Process(pid)
        if p.name() == "python.exe":
            for index, command in enumerate(p.cmdline(), start=0):
                if "--axle_ip_address" in command:
                    cmd_axle_ip = p.cmdline()[index + 1]
                    if cmd_axle_ip == axle_ip:
                        print("axle Ip =" + command)
                        return cmd_axle_ip

                if "--front_ip_address" in command:
                    cmd_front_ip = p.cmdline()[index + 1]
                    if cmd_front_ip == front_ip:
                        print("front Ip =" + command)
                        return cmd_front_ip

                if "--back_ip_address" in command:
                    cmd_back_ip = p.cmdline()[index + 1]
                    if cmd_back_ip == back_ip:
                        print("back Ip =" + command)
                        return cmd_back_ip
    return None


def kill_process(axle_ip):
    for pid in psutil.pids():
        p = psutil.Process(pid)
        if p.name() == "python.exe":
            for index, command in enumerate(p.cmdline(), start=0):
                if "--axle_ip_address" in command:
                    cmd_axle_ip = p.cmdline()[index + 1]
                    if cmd_axle_ip == axle_ip:
                        print("axle Ip =" + command)
                        # terminate running process
                        p.terminate()
                        return cmd_axle_ip

    return None


@app.route("/run-license-detection", methods=['POST'])
def plate_detection():
    data = request.get_json(force=True)
    command, axle_ip_address, front_ip_address, back_ip_address = prepare_command(data, False)
    # check  if process no already running
    response = verification(axle_ip_address, front_ip_address, back_ip_address)
    if response is None:
        run_detection(command)
        json_response(status=200, message="Plate Detection running on")
        return "Plate Detection running on", 200
    else:
        json_response(status=500, message="Plate Detection is ALREADY running on" + response)
        return "Plate Detection is ALREADY running on" + response, 500


@app.route('/hi')
def hello_world():
    run_detection("python  C:/Apache24/htdocs/api/vtmsdetectionsystem/testCamera.py")
    return 'Hello Flask under Apache!'


@app.route("/run-axle-counter", methods=['POST'])
def axle_counter():
    data = request.get_json(force=True)
    command, axle_ip_address, front_ip_address, back_ip_address = prepare_command(data, True)
    # check  if process no already running
    response = verification(axle_ip_address, front_ip_address, back_ip_address)
    if response is None:
        run_detection(command)
        json_response(status=200, message="Axle Counter running on")
        return "Plate Detection running on", 200
    else:
        json_response(status=500, message="Axle Counter is ALREADY running on" + response)
        return "Axle Counter is ALREADY running on" + response, 500


@app.route("/capture-images", methods=['POST'])
def capture_images():
    data = request.get_json(force=True)
    command, axle_ip_address, front_ip_address, back_ip_address = prepare_command(data, True, True)
    # check  if process no already running
    response = verification(axle_ip_address, front_ip_address, back_ip_address)
    if response is None:
        run_detection(command)
        json_response(status=200, message="image Capture running on")
        return "Image Capture running on", 200
    else:
        json_response(status=500, message="image Captureis ALREADY running on" + response)
        return "image Capture is ALREADY running on" + response, 500


@app.route("/kill-process", methods=['POST'])
def kill_process():
    data = request.get_json(force=True)
    axle_ip_address = data['axleIp']  # path to the media to be used for detection
    response = kill_process(axle_ip_address)
    if response is None:
        json_response(status=200, message="No Detection is running this Ip Address")
        return "No Detection is running this Ip Address", 200
    else:
        json_response(status=200, message="No Detection is running on " + response + " has been stopped")
        return "No Detection is running on " + response + " has been stopped", 200


## picture detection method

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
