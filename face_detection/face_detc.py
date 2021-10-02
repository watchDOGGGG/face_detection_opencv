import cv2

# load some pre-trained data for open cv
trained_face_data = cv2.CascadeClassifier('face_detection\haarcascade_frontalface.xml') # cv2.data.haarcascades +

# chose image to detect face in it
img = cv2.imread("face_detection\me.jpg")
#convert to gray_scale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Detect faces
faces = trained_face_data.detectMultiScale(grayscaled_img)

# draw rectangles around the faces
for (x,y,w,h) in faces:
    cv2.rectangle(grayscaled_img, (x,y), (x+w , y+h), (0,225,0) ,2)

# show the image
cv2.imshow('prince face detector', grayscaled_img)

# pause execution of code
cv2.waitKey()



print('code completed')