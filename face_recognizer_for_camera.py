import cv2

# load cascade classifier training file for haarcascade
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

# load the already trained face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trained_recognizer.yml')

# set the lable collection
label = ["", "Boris", "Otis", "Siliang"] 

# open the camera of laptop
lap_camera = cv2.VideoCapture(0)
while True:
    # read the current frame from camera
    sucess, frame = lap_camera.read()
    # convert the color frame to gray frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect all faces in the current frame
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    # recognize all the faces in the current frame
    for (x,y,w,h) in faces:
        # predict the label of each face
        predict_label, confidence = face_recognizer.predict(gray_frame[y:y+h, x:x+w])
        face_name = label[predict_label]
        # draw rectangle for each detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2, cv2.LINE_8, 0)
        # show the corresponding name for the predicted label
        if confidence > 50:
            cv2.putText(frame, face_name, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)    
    
    cv2.imshow("frame",frame)
    
    # keep playing continual frame
    k=cv2.waitKey(1)
    if k == 27:
        # quit by press 'Esc'
        cv2.destroyAllWindows()
        break
    elif k==ord("s"):
        # save current frame by press 's'
        cv2.imwrite("ssl2.jpg",faces)
        cv2.destroyAllWindows()
        break

# close the camera
lap_camera.release()
cv2.destroyAllWindows()

