import cv2
from random import randrange
haacasade_train_face = cv2.CascadeClassifier("face_detection\haarcascade_frontalface.xml")

video_cp = cv2.VideoCapture(0)

# throw error is webcam fail
if not video_cp.isOpened():
    raise IOError("cant open video")

# if video opens get frame and resize to fit browser
while True:
    ret, frame = video_cp.read()
    frame = cv2.resize(frame, None, fx =1, fy=1, interpolation=cv2.INTER_AREA)
    
    # set frame to grayscale
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect_face_coordinates
    detect_face_coordinates = haacasade_train_face.detectMultiScale(frame)
    
    for x,y,w,h in detect_face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w, y+h), (randrange(200), randrange(250), randrange(200)),2)
    
    cv2.imshow('Input', frame)
    
    c = cv2.waitKey(1)  
    if c == 27:
        break 