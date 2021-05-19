import os

import cv2
import pytesseract
from datetime import datetime
from core.config import cfg
from core.utils import read_class_names


# function to count objects, can return total classes or count per class
def count_objects(data, by_class=False, allowed_classes=list(read_class_names(cfg.YOLO.CLASSES).values())):
    boxes, scores, classes, num_objects = data

    # create dictionary to hold count of objects
    counts = dict()

    # if by_class = True then count objects per class
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)

        # loop through total number of objects found
        for i in range(num_objects):
            # grab class index and convert into corresponding class name
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
            else:
                continue

    # else count total objects found
    else:
        counts['total object'] = num_objects

    return counts


# function for cropping each detection and saving as new image
def crop_objects(img, data, path, allowed_classes, ip_address):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.LICENSE_PLATE)
    # create dictionary to hold count of objects for image name
    counts = dict()
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            # crop detection from image (take an additional 5 pixels around all edges)
            cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            # construct image name and join it to path for saving crop properly
            img_name = class_name + '_' + ip_address + '.png'
            img_path = os.path.join(path, img_name)
            # save image
            cv2.imwrite(img_path, cropped_img)
            return img_name
        else:
            continue


# function for cropping each detection and saving as new image
def write_360_img(ip_addresses, credentials):
    img_locations = {}
    username = credentials['username']
    password = credentials['password']
    # loop through all values in ip_addresses, capture images, store images and finally send back locations
    for k, v in ip_addresses.items():
        # the side of the vehicle is the k in the dictionary
        # e.g {BACK_LEFT : 192.1.2.3,BACK_RIGHT : 192.4.5.6}
        side = k
        ip_address = v
        #  start capture stream
        # construct url
        url = 'rtsp://' + username + ':' + password + '@' + ip_address + ':554/Streaming/Channels/1'
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
                img = frame
                break

        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        # cv2.destroyAllWindows()
        # path where the image is to be saved
        path = os.path.join('C:\\app_upload\\uploads\\tempVehicleDetails\\', '360_IMG', ip_address)
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        img_name = side + '_' + ip_address + '.png'
        img_path = os.path.join(path, img_name)
        # save image
        cv2.imwrite(img_path, img)

        # create a map for each saved image and its location and send back for saving
        img_locations.update({side: os.path.join(ip_address, img_name)})

    return img_locations


# function to run general Tesseract OCR on any detections
def ocr(img, data):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.LICENSE_PLATE)
    for i in range(num_objects):
        # get class name for detection
        class_index = int(classes[i])
        class_name = class_names[class_index]
        # separate coordinates from box
        xmin, ymin, xmax, ymax = boxes[i]
        # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
        box = img[int(ymin) - 5:int(ymax) + 5, int(xmin) - 5:int(xmax) + 5]
        # grayscale region within bounding box
        gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
        # threshold the image using Otsus method to preprocess for tesseract
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # perform a median blur to smooth image slightly
        blur = cv2.medianBlur(thresh, 3)
        # resize image to double the original size as tesseract does better with certain text size
        blur = cv2.resize(blur, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # run tesseract and convert image text to string
        try:
            text = pytesseract.image_to_string(blur, config='--psm 11 --oem 3')
            print("Class: {}, Text Extracted: {}".format(class_name, text))
        except:
            text = None
