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
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('is_plate', False, 'checks whether its license plate or axle count')
flags.DEFINE_string('gate_id', '0', 'id of gate detection is being run on ?')
flags.DEFINE_string('transaction', '', 'transaction reference for the exit vehicle?')

# axle count data
flags.DEFINE_string('axle_weights', './checkpoints/yolov4-tiny-axle-416', 'path to weights file')
flags.DEFINE_string('axle_ip_address', '192.168.8.100', 'ip address of the device capturing')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_string('video', '0', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
# plate data
flags.DEFINE_string('plate_weights', './checkpoints/yolov4-license-plate-416',
                    'path to weights file')
flags.DEFINE_string('front_ip_address', '192.168.8.101', 'ip address of the device capturing')
flags.DEFINE_string('back_ip_address', '192.168.8.101', 'ip address of the device capturing')
# define 360 ip addresses
flags.DEFINE_string('back_left', '192.168.8.8', 'ip address of the device capturing back left')
flags.DEFINE_string('back_right', '192.168.8.8', 'ip address of the device capturing back right')
flags.DEFINE_string('front_left', '192.168.8.8', 'ip address of the device capturing front left')
flags.DEFINE_string('front_right', '192.168.8.8', 'ip address of the device capturing front right')
flags.DEFINE_string('username', 'admin', ' user name used to log into the 360 cameras')
flags.DEFINE_string('front_password', 'Admin1234', ' password used to log into the 360 cameras')
flags.DEFINE_string('back_password', 'Admin1234', ' password used to log into the 360 cameras')
flags.DEFINE_string('password', 'Admin1234', ' password used to log into the 360 cameras')
flags.DEFINE_string('front_user_name', 'admin', ' user name used to log into the 360 cameras')
flags.DEFINE_string('back_user_name', 'admin', ' user name used to log into the 360 cameras')
flags.DEFINE_string('port', '554', ' port to be used to access the cameras')

path = os.path.dirname(os.path.abspath(__file__))


def main(_argv):
    # load tflite model if flag is set
    axle_saved_model_loaded = tf.saved_model.load(FLAGS.axle_weights, tags=[tag_constants.SERVING])
    axle_infer = axle_saved_model_loaded.signatures['serving_default']

    # load model
    plate_saved_model_loaded = tf.saved_model.load(FLAGS.plate_weights, tags=[tag_constants.SERVING])
    plate_infer = plate_saved_model_loaded.signatures['serving_default']

    if FLAGS.is_plate:
        detect_plate(FLAGS, plate_infer)
    else:
        count_axles(FLAGS, axle_infer, plate_infer)


def detect_plate(_argv, infer):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    input_size = 416
    # ip_address = FLAGS.ip_address
    transaction = FLAGS.transaction
    # create 360 dictionary
    ip_addresses = {"BACK_LEFT": FLAGS.back_left, "BACK_RIGHT": FLAGS.back_right, "FRONT_LEFT": FLAGS.front_left,
                    "FRONT_RIGHT": FLAGS.front_right}
    credentials = {"username": FLAGS.username, "password": FLAGS.password}
    # loop through images in list and run Yolov4 model on each
    # f_image, b_image = get_frames(FLAGS.front_ip_address, FLAGS.back_ip_address, FLAGS.front_password,
    #                               FLAGS.back_password, FLAGS.front_user_name,
    #                               FLAGS.back_user_name, FLAGS.port)
    # loop through images in list and run Yolov4 model on each
    f_image = cv2.imread('./data/images/car9.png')
    b_image = cv2.imread('./data/images/test_car11.png')
    all_images = [f_image, b_image]

    for count, image in enumerate(all_images, start=0):
        original_image = image
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.

        # get image name by using split method

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        # infer =
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
        bboxes = utils.format_plate_boxes(boxes.numpy()[0], original_h, original_w)
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
        img_name = crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path,
                                allowed_classes, ip_address)

        final_img_path = os.path.join(ip_address, img_name)
        image, plate_number = utils.draw_bbox(original_image, pred_bbox, False, allowed_classes=allowed_classes,
                                              read_plate=True)
        # show center y
        # cv2.line(original_image, (0, int(((bbox[1]) + (bbox[3])) / 2)),
        #          (original_w, int(((bbox[1]) + (bbox[3])) / 2)), (0, 255, 255), thickness=1)
        img_locations = ""
        gate_id = FLAGS.gate_id
        # check if its not a failed recognition
        # if plate_number != "":
        # capture 360 images before exiting, but only if front plate
        if count == 0:
            img_locations = write_360_img(ip_addresses, credentials, '')
            vehicle_side = "FRONT"
        else:
            vehicle_side = "BACK"
        for plate in plate_number:
            return_detected_plate_details(final_img_path, plate, vehicle_side, ip_address, gate_id, img_locations,
                                      transaction)
            print("sent" + plate)

        image = Image.fromarray(image.astype(np.uint8))
        if not False:
            image.show()
        # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        # cv2.imwrite(FLAGS.output + 'detection.png', image)

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


def count_axles(_argv, infer, plate_infer):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = path + '/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)
    counter = []
    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    frame_counter = 500
    frame_num = 0
    total_count = 0
    initial_time = time.time()

    # while video is running
    while True:
        return_value, original_frame = vid.read()
        if return_value:
            # cv2.rectangle(original_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Crop image
            # cropped_frame = original_frame[y1:y2, x1:x2]
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(original_frame)
        else:
            print('Video has ended or failed, try a different video format!')

            vid = cv2.VideoCapture(video_path)
            return_value, original_frame = vid.read()
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(original_frame)

        frame_num += 1
        print('Frame #: ', frame_num)
        frame_size = original_frame.shape[:2]
        image_data = cv2.resize(original_frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # infer =
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = original_frame.shape
        bboxes = utils.format_axle_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.AXLE)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        # allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(original_frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        current_count = int(0)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(original_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(original_frame, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            cv2.putText(original_frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0,
                        0.75, (255, 255, 255), 2)

            # lets count our axles
            # create a line in the middle of the frame for the license plate recognition the line will be horizonta;
            cv2.line(original_frame, (int((original_w / 2) - 100), int(3 * original_h / 7)),
                     (int((original_w / 2) + 100), int(3 * original_h / 7)), (0, 255, 0),
                     thickness=2)

            center_x = int(((bbox[0]) + (bbox[2])) / 2)
            center_y = int(((bbox[1]) + (bbox[3])) / 2)

            if center_x <= int(3 * original_w / 6 + original_w / 90) and center_x >= int(
                    3 * original_w / 6 - original_w / 90) and center_y >= int(3 * original_h / 7):
                print("something has crossed the line")
                current_count += 1
                if int(track.track_id) not in counter:
                    counter.append(int(track.track_id))
                    initial_time = time.time() - 20
                    print("different axle")
            else:
                frame_counter = abs(frame_counter) - 1

            total_count = len(set(counter))
            cv2.putText(original_frame, "Total Axle Count: " + str(total_count), (0, 130), 0, 1, (0, 0, 255), 2)
            cv2.putText(original_frame, "Current Axle Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)

        print("frame count =" + str(frame_counter))
        current_time = time.time() - initial_time

        if current_time >= 30 and total_count > 1:
            # run plate detection once vehicle in desired zone

            print("Total count =" + str(total_count))
            return_detected_axle_details(total_count, FLAGS.axle_ip_address, FLAGS.gate_id)
            counter = []
            total_count = int(0)
            # initialize the timer
            initial_time = time.time()
            # pause the program
            # time.sleep(50)
            # re-Initialize tracker for new vehicle
            tracker = Tracker(metric)
            detect_plate(FLAGS, plate_infer)

        # calculate frames per second of running detections
        t = (time.time() - start_time)
        if t != 0:
            fps = 1.0 / t
            print("FPS: %.2f" % fps)
        print("Time: %.2f" % current_time)
        result = np.asarray(original_frame)
        result = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            # cv2.imshow("cropped_frame", cropped_frame)
            cv2.imshow("original_frame", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
