import os

# comment out below line to enable tensorflow outputs
from detection_response import return_detected_plate_details

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
# changed the import to custom vtms util
import core.vtUtil as utils
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


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    # get video name by using split method
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None
    if True:
     Fvalue, frame = vid.read()
    # set region of interest
    # r = cv2.selectROI(frame)
    # set region of interest
    r = (602, 377, 538, 239)
    x1 = int(r[0])
    x2 = int(r[0] + r[2])
    y1 = int(r[1])
    y2 = int(r[1] + r[3])

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    while True:
        return_value, original_frame = vid.read()

        if return_value:
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Crop image
            cropped_frame = original_frame[y1:y2, x1:x2]

            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
            image = Image.fromarray(cropped_frame)
        else:
            print('Video has ended or failed, try a different video format!')

            vid = cv2.VideoCapture(video_path)
            return_value, original_frame = vid.read()
            # set region of interest
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Crop image
            cropped_frame = original_frame[y1:y2, x1:x2]
            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
            image = Image.fromarray(cropped_frame)

        frame_size = cropped_frame.shape[:2]
        image_data = cv2.resize(cropped_frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

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

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = cropped_frame.shape

        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
        bbox = bboxes[0]
        # create a line in the middle of the frame for the license plate recognition the line will be horizontal;
        # cv2.line(frame, (0, int(3 * original_h / 6)), (original_w, int(3 * original_h / 6)), (0, 255, 0), thickness=1)
        # cv2.line(frame, (0, int(3 * original_h / 6 + original_h / 80)),
        #          (original_w, int(3 * original_h / 6 + original_h / 80)), (0, 255, 0), thickness=1)
        # cv2.line(frame, (0, int(3 * original_h / 6 - original_h / 80)),
        #          (original_w, int(3 * original_h / 6 - original_h / 80)), (0, 255, 0), thickness=1)
        # cv2.line(frame, (0, int(((bbox[1]) + (bbox[3])) / 2)),
        #          (original_w, int(((bbox[1]) + (bbox[3])) / 2)), (0, 255, 255), thickness=1)
        # get center_y
        center_y = int(((bbox[1]) + (bbox[3])) / 2)
        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.LICENSE_PLATE)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        if center_y <= int(3 * original_h / 6 + original_h / 100) and center_y >= int(
                3 * original_h / 6 - original_h / 100):
            print("something has crossed the line")

            # custom allowed classes (uncomment line below to allow detections for only people)
            # allowed_classes = ['person']
            # initialise path
            final_path = ""
            # if crop flag is enabled, crop each detection and save it as new image
            if FLAGS.crop:
                crop_rate = 150  # capture images every so many frames (ex. crop photos every 150 frames)
                crop_path = os.path.join(os.getcwd(), 'detections', 'crop', video_name)
                try:
                    os.mkdir(crop_path)
                except FileExistsError:
                    pass
                if frame_num % crop_rate == 0:
                    final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
                    try:
                        os.mkdir(final_path)
                    except FileExistsError:
                        pass
                    crop_objects(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
                else:
                    pass


            image, plate_number = utils.draw_bbox(cropped_frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes,
                                                  read_plate=FLAGS.plate)
            # show center y
            cv2.line(cropped_frame, (0, int(((bbox[1]) + (bbox[3])) / 2)),
                     (original_w, int(((bbox[1]) + (bbox[3])) / 2)), (0, 255, 255), thickness=1)

            vehicle_side = FLAGS.vehicle_side
            ## try to capture path and plate number
            # return_detected_plate_details(final_path, plate_number, vehicle_side)

        else:
            image, plate_number = utils.draw_bbox(cropped_frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes,
                                                  read_plate=False)
            # show center y
            cv2.line(cropped_frame, (0, int(((bbox[1]) + (bbox[3])) / 2)), (original_w, int(((bbox[1]) + (bbox[3])) / 2)),
                     (0, 255, 255), thickness=1)

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            # cv2.imshow("result", result)
            cv2.imshow("Full Frame View", original_frame)

        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
