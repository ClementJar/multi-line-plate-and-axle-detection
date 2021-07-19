import os
# comment out below line to enable tensorflow outputs
from detection_response import return_detected_plate_details

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

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
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('ip_address', '192.168.0.1', 'ip address of the device capturing')
flags.DEFINE_string('vehicle_side', 'front', 'back or front of vehicle ?')
flags.DEFINE_string('gate_id', '0', 'id of gate detection is being run on ?')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')
# define 360 ip addresses
flags.DEFINE_string('back_left', '192.168.0.1', 'ip address of the device capturing back left')
flags.DEFINE_string('back_right', '192.168.0.1', 'ip address of the device capturing back right')
flags.DEFINE_string('front_left', '192.168.0.1', 'ip address of the device capturing front left')
flags.DEFINE_string('front_right', '192.168.0.1', 'ip address of the device capturing front right')
flags.DEFINE_string('username', 'admin', ' user name used to log into the 360 cameras')
flags.DEFINE_string('password', 'Admin1234', ' password used to log into the 360 cameras')

def detect_plate(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    ip_address = FLAGS.ip_address

    # load model
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])

    # create 360 dictionary
    ip_addresses = {"BACK_LEFT": FLAGS.back_left, "BACK_RIGHT": FLAGS.back_right, "FRONT_LEFT": FLAGS.front_left,
                    "FRONT_RIGHT": FLAGS.front_right}
    credentials = {"username": FLAGS.username, "password": FLAGS.password}
    # loop through images in list and run Yolov4 model on each

    #  start capture stream
    # construct url
    url = video_path
    # define a video capture object
    # vid = cv2.VideoCapture("./data/video/plate4.mp4")
    vid = cv2.VideoCapture(url)

    while True:
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        if ret:
            # Display the resulting frame
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)
            original_image = frame

            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            image_data = cv2.resize(original_image, (input_size, input_size))
            image_data = image_data / 255.

            # get image name by using split method

            images_data = []
            for i in range(1):
                images_data.append(image_data)
            images_data = np.asarray(images_data).astype(np.float32)

            if FLAGS.framework == 'tflite':
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], images_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                infer = saved_model_loaded.signatures['serving_default']
                batch_data = tf.constant(images_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            # run non max suppression on detections
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
            original_h, original_w, _ = original_image.shape
            bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
            bbox = bboxes[0]

            # hold all detection data in one variable
            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.LICENSE_PLATE)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())
            # center_y = int(((bbox[1]) + (bbox[3])) / 2)

            # if int(3 * original_h / 6 + original_h / 200) >= center_y >= int(
            #         3 * original_h / 6 - original_h / 200):
            #     print("something has crossed the line")

            # custom allowed classes (uncomment line below to allow detections for only people)
            # allowed_classes = ['person']
            # initialise path
            ip_address = FLAGS.ip_address
            img_name = ""
            # if crop flag is enabled, crop each detection and save it as new image
            if FLAGS.crop:
                crop_path = os.path.join('C:\\app_upload\\uploads\\tempVehicleDetails\\', ip_address)
                try:
                    os.mkdir(crop_path)
                except FileExistsError:
                    pass
                img_name = crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path,
                                        allowed_classes, ip_address)

            final_img_path = os.path.join(ip_address, img_name)
            image, plate_number = utils.draw_bbox(original_image, pred_bbox, FLAGS.info, allowed_classes=allowed_classes,
                                                  read_plate=FLAGS.plate)
            # show center y
            # cv2.line(original_image, (0, int(((bbox[1]) + (bbox[3])) / 2)),
            #          (original_w, int(((bbox[1]) + (bbox[3])) / 2)), (0, 255, 255), thickness=1)

            vehicle_side = FLAGS.vehicle_side
            gate_id = FLAGS.gate_id
            # check if its not a failed recognition
            if plate_number != "":
                # capture 360 images before exiting, but only if front plate
                if vehicle_side == 'FRONT':
                    img_locations = write_360_img(ip_addresses, credentials)

                # return_detected_plate_details(final_img_path, plate_number, vehicle_side, ip_address, gate_id, img_locations)

            image = Image.fromarray(image.astype(np.uint8))
            if not FLAGS.dont_show:
                image.show()
            # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            # cv2.imwrite(FLAGS.output + 'detection.png', image)

        break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows


if __name__ == '__main__':
    try:
        app.run(detect_plate)
    except SystemExit:
        pass
