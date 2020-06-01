# The main lib used here is OpenCV which is a huge open-source library for computer vision, machine learning, and image processing.
import cv2

# Object Detection using Haar feature-based cascade classifiers is an object detection method. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images by superimposing predefined patterns over face segments and are used as XML files. Face, eye and smile haar-cascades are used here.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Here above statement will create a VideoCapture object and will return it, as the value is stored in varriable. Since a video is a stream of picture frames, using this VideoCapture object we will access the camera and retrieve each frame and display it on the screen
video_capture = cv2.VideoCapture(0)


while True:
    # Captures video_capture frame by frame
    _, frame = video_capture.read()

    # Here the frame is turned to grey scale image as haar-cascades work better on them
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # here face is detected, where 1.3 is the scaling factor, and 5 is the number of nearest neighbors.
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # this loop will draw a rectangle on face in case of detection. The cv2.rectangle method takes 5 arguments -- Image, Top-left corner coordinates, Bottom-right corner coordinates, Color (in BGR format), Line width. Same logic for eyes and smile.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey),
                          ((ex + ew), (ey + eh)), (0, 0, 255), 2)

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy),
                          ((sx + sw), (sy + sh)), (0, 255, 0), 2)

        # As the detectMultiScale method of CascadeClassifier is returning numpy n dimensional array, here we check if both eyes and smile is detected by checking the array length and save the frame(picture) if condition is satisfied․ So the picture will be taken only when both eyes and smile is detected.
        if len(eyes) > 1 and len(smiles) > 0:
            cv2.imwrite("SmilePicture.jpg", frame)

    # Displays the result on camera feed
    cv2.imshow('Video', frame)


# The control breaks once q key is pressed

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
