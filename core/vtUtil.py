import colorsys
import random
import re
# from imutils import contours
import cv2
import numpy as np
import pytesseract
import tensorflow as tf
from core.config import cfg


def recognize_vt_plate(img, coords):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    box = img[int(ymin) - 5:int(ymax) + 5, int(xmin) - 5:int(xmax) + 5]
    # grayscale region within bounding box
    gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
    # resize image to three times as large as original for better readability
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("gray", gray)
    # cv2.waitKey(0)
    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow("GaussianBlur", blur)
    # cv2.waitKey(0)
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # cv2.imshow("Otsu Threshold", thresh)
    # cv2.waitKey(0)
    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # apply dilation to make regions more clear
    open_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rect_kern)

    # cv2.imshow("Dilation", dilation)
    # open_img = cv2.bitwise_not(open_img)
    # cv2.waitKey(0)
    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv2.findContours(open_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(open_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    im2 = open_img.copy()
    #################################################
    # Will need to improve recognition process
    ##########################################################
    # create copy of gray image

    im2 = cv2.dilate(im2, rect_kern)


    # perform another blur on character region
    # im2 = cv2.bitwise_not(im2)
    rect = cv2.medianBlur(im2, 5)

    # perform another blur on character region
    rect = cv2.medianBlur(rect, 5)
    # rect = cv2.bitwise_not(rect)
    # perform another blur on character region
    rect = cv2.medianBlur(rect, 7)
    # perform another blur on character region
    # cv2.imshow("rect", rect)
    # cv2.waitKey(0)
    # create blank string to hold license plate number
    plate_num = ""
    try:
        # run through ocr
        text = pytesseract.image_to_string(rect,
                                           config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 -l eng --oem 3')
        # clean tesseract text by removing any unwanted blank spaces
        clean_text = re.sub('[\W_]+', '', text)
        plate_num = clean_text
        print(plate_num + "1")
        # cv2.imshow("rect2", rect)
        # cv2.waitKey(0)
        if not len(plate_num) > 4:
            # run through ocr with bitwise_not applied
            text = pytesseract.image_to_string(cv2.medianBlur(rect, 5),
                                               config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 -l eng --oem 3')
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num = clean_text
            print(plate_num + "2")

        if not len(plate_num) > 4:
            # run through ocr with different --psm
            text = pytesseract.image_to_string(rect,
                                               config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6 -l eng --oem 3')
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num = clean_text
            print(plate_num + "3")

        if not len(plate_num) > 4:
            # run through ocr with different --psm and bitwise_not applied
            text = pytesseract.image_to_string(rect,
                                               config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6 -l eng --oem 3')
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num = clean_text
            print(plate_num + "4")

        if not len(plate_num) > 4:
            # run through ocr with different --psm
            text = pytesseract.image_to_string(rect,
                                               config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 4 -l eng --oem 3')
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num = clean_text
            print(plate_num + "5")

        if not len(plate_num) > 4:
            # run through ocr with different --psm and bitwise_not applied
            text = pytesseract.image_to_string(cv2.medianBlur(rect, 5),
                                               config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 4 -l eng --oem 3')
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num = clean_text
            print(plate_num + "6")

        if not len(plate_num) > 4:
            # run through ocr with different --psm
            text = pytesseract.image_to_string(rect,
                                               config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7 -l eng --oem 3')
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num = clean_text
            print(plate_num + "7")

        if not len(plate_num) > 4:
            # run through ocr with different --psm and bitwise_not applied
            text = pytesseract.image_to_string(cv2.medianBlur(rect, 5),
                                               config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7 -l eng --oem 3')
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num = clean_text
            print(plate_num + "8")

    except:
        text = None
    if plate_num is not None:
        print("License Plate #: ", plate_num)
    # cv2.imshow("Character's Segmented", im2)
    # cv2.waitKey(0)
    return plate_num
