import cv2
import queue
import time
import threading

q = queue.Queue()


def Receive():
    print("start Reveive")
    # cap = cv2.VideoCapture("rtsp://cbuibic:admin1234@192.168.8.65:554/Streaming/Channels/101")
    cap = cv2.VideoCapture("rtsp://admin:Admin1234@192.168.8.64:554/Streaming/Channels/1")
    # cap = cv2.VideoCapture("http://admin:Admin1234@192.168.1.65:80/Streaming/Channels/1")
    # cap = cv2.VideoCapture("http://admin:Admin1234@192.168.1.64:80/Streaming/channels/1")
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)


def Display():
    print("Start Displaying")
    while True:
        if q.empty() != True:
            frame = q.get()
            r = (602, 377, 538, 239)
            x1 = int(r[0])
            x2 = int(r[0] + r[2])
            y1 = int(r[1])
            y2 = int(r[1] + r[3])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Crop image
            imCrop = frame[y1:y2, x1:x2]
            # blow up
            # imCrop = cv2.resize(imCrop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            # # Display cropped image
            cv2.imshow("full frame", frame)
            # cv2.imshow("Image", imCrop)

            # cv2.waitKey(0)
            # cv2.imshow("frame1", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()
