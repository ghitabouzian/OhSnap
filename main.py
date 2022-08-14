from cv2 import cv2
import cvzone
captureVideo =  cv2.VideoCapture(0) #capturing video
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #to detect face on grayscale
overlay = cv2.imread('native.png' , cv2.IMREAD_UNCHANGED)

while True: #continuous loop
    _, frame = captureVideo.read() #read from video capture
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # grayscale makes it easier to detect face
    faces = cascade.detectMultiScale(grayscale) #detects face in the grayscale
    for (x, y, w, h) in faces: #for loop to detect faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        overlayResize = cv2.resize(overlay, (int(w*1.5), int(h*1.5))) #resizing our overlay
        frame = cvzone.overlayPNG(frame, overlayResize, [x-45, y-75])

    cv2.imshow('Oh Snap', frame)
    if cv2.waitKey(10) == ord('q') : #stop video when user clicks on q key
        break




