import cv2
thermal_path = "rtsp://admin:Admin1234@192.168.1.64:554/Streaming/Channels/2"
thermal_capture = cv2.VideoCapture(thermal_path)
while True:

    ret, thermal_frame = thermal_capture.read(0)
    if not ret:
        thermal_capture.release()
        thermal_capture = cv2.VideoCapture(thermal_path)
        print('Found error; rebuilding stream')

    #do a lot of things

thermal_capture.release()
cv2.destroyAllWindows()