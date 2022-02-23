import os
# comment out below line to enable tensorflow outputs
from detection_response import return_detected_plate_details, return_detected_axle_details

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import psutil
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import time
from core.config import cfg
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# General
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('is_plate', False, 'checks whether its license plate or axle count')
flags.DEFINE_string('gate_id', '0', 'id of gate detection is being run on ?')
flags.DEFINE_string('transaction', '', 'transaction reference for the exit vehicle?')

# axle count data
flags.DEFINE_string('axle_weights', './checkpoints/yolov4-tiny-axle-416', 'path to weights file')
flags.DEFINE_string('axle_ip_address', '192.168.0.1', 'ip address of the device capturing')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_string('video', './data/video/axle1.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
# plate data
flags.DEFINE_string('plate_weights', './checkpoints/yolov4-license-plate-416',
                    'path to weights file')
flags.DEFINE_string('front_ip_address', '192.168.8.100', 'ip address of the device capturing')
flags.DEFINE_string('back_ip_address', '192.168.8.101', 'ip address of the device capturing')
# define 360 ip addresses
flags.DEFINE_string('back_left', '192.168.8.100', 'ip address of the device capturing back left')
flags.DEFINE_string('back_right', '192.168.8.100', 'ip address of the device capturing back right')
flags.DEFINE_string('front_left', '192.168.8.100', 'ip address of the device capturing front left')
flags.DEFINE_string('front_right', '192.168.8.100', 'ip address of the device capturing front right')
flags.DEFINE_string('username', 'admin', ' user name used to log into the 360 cameras')
flags.DEFINE_string('front_password', 'Admin1234', ' password used to log into the 360 cameras')
flags.DEFINE_string('back_password', 'Admin1234', ' password used to log into the 360 cameras')
flags.DEFINE_string('password', 'Admin1234', ' password used to log into the 360 cameras')
flags.DEFINE_string('front_user_name', 'admin', ' user name used to log into the 360 cameras')
flags.DEFINE_string('back_user_name', 'admin', ' user name used to log into the 360 cameras')
flags.DEFINE_string('port', '554', ' port to be used to access the cameras')

path = os.path.dirname(os.path.abspath(__file__))

def main(_argv):

    # ip_address = FLAGS.ip_address
    transaction = FLAGS.transaction
    # create 360 dictionary
    ip_addresses = {"BACK_LEFT": FLAGS.back_left, "BACK_RIGHT": FLAGS.back_right, "FRONT_LEFT": FLAGS.front_left,
                    "FRONT_RIGHT": FLAGS.front_right}
    credentials = {"username": FLAGS.username, "password": FLAGS.password}
    # loop through images in list and run Yolov4 model on each
    f_image, b_image = get_frames(FLAGS.front_ip_address,
                                  FLAGS.back_ip_address,
                                  FLAGS.front_password,
                                  FLAGS.back_password,
                                  FLAGS.front_user_name,
                                  FLAGS.back_user_name,
                                  FLAGS.port)
    # loop through images in list and run Yolov4 model on each
    all_images = [f_image, b_image]

    for count, image in enumerate(all_images, start=0):
        original_image = image

        if count == 0:
            ip_address = FLAGS.front_ip_address
        else:
            ip_address = FLAGS.back_ip_address

        img_name = ""
        # if crop flag is enabled, crop each detection and save it as new image
        crop_path = os.path.join('C:\\app_upload\\vtms\\uploads\\tempVehicleDetails\\LPNR\\', ip_address)
        try:
            os.makedirs(crop_path)
        except FileExistsError:
            pass
        img_name = save_images(original_image, crop_path, ip_address, transaction)

        final_img_path = os.path.join(ip_address, img_name)
        # show center y
        # cv2.line(original_image, (0, int(((bbox[1]) + (bbox[3])) / 2)),
        #          (original_w, int(((bbox[1]) + (bbox[3])) / 2)), (0, 255, 255), thickness=1)
        img_locations = ""
        plate_number = ""
        gate_id = FLAGS.gate_id
        # check if its not a failed recognition
        # if plate_number != "":
        # capture 360 images before exiting, but only if front plate
        if count == 0:
            img_locations = write_360_img(ip_addresses, credentials, transaction)
            vehicle_side = "FRONT"
        else:
            vehicle_side = "BACK"

        return_detected_plate_details(final_img_path, plate_number, vehicle_side, ip_address, gate_id, img_locations, transaction)


    # Destroy all the windows


def get_frames(f_ip, b_ip, f_pass, b_pass, f_name, b_name, port):
    #  start capture stream
    # construct url
    f_image = None
    b_image = None
    # define a video capture object
    # vid = cv2.VideoCapture("./data/video/
    for ip in (f_ip, b_ip):
        camera_url = ""

        if ip == f_ip:
            camera_url = "rtsp://" + f_name + ":" + f_pass + "@" + f_ip + ":" + port + "/Streaming/Channels/1"
        else:
            # camera_url = "rtsp://" + b_name + ":" + b_pass + "@" + b_ip + ":" + port + "/Streaming/Channels/1"
            camera_url = "rtsp://" + b_name + ":" + b_pass + "@" + f_ip + ":" + port + "/Streaming/Channels/1"

        vid = cv2.VideoCapture(camera_url)

        while True:
            # Capture the video frame
            # by frame
            ret, frame = vid.read()

            if ret:
                # Display the resulting frame
                # cv2.imshow('frame', frame)
                # cv2.waitKey(0)
                if ip == f_ip:
                    f_image = frame
                    b_image = frame
                else:
                    b_image = frame

                break
            else:
                print("No feed")
                return None

    # After the loop release the cap object
    # cv2.imshow('b_image', b_image)
    # cv2.imshow('f_image', f_image)
    # cv2.waitKey(0)
    vid.release()

    return f_image, b_image

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
