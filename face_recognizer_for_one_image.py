import cv2

def recognize_all_faces_in_one_image(test_image_path):
    # input: the directory of test image
    # output: the test image with rectangles around faces
    
    # load cascade classifier training file for haarcascade
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

    # load the already trained face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('trained_recognizer.yml')

    # set the lable collection
    label = ["", "Boris", "Otis", "Siliang"] 

    # read the test image
    test_image = cv2.imread(test_image_path)
    # convert to gray image for face detection
    gray_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    # detect all faces in the current frame, this line below return locations of all faces in the image
    faces = face_detector.detectMultiScale(gray_test_image, scaleFactor=1.1, minNeighbors=5)

    for (x,y,w,h) in faces:
        # predict the label
        predict_label, confidence = face_recognizer.predict(gray_test_image[y:y+h, x:x+w])
        # read the name of specific label
        face_name = label[predict_label]
        # draw rectangle  around the face
        cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 0, 255), 2, cv2.LINE_8, 0)
        # draw the name
        if confidence > 50:
            cv2.putText(test_image, face_name, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        else:
            cv2.putText(test_image, "Unknown", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)  

    # draw the test image with rectangles marked around each face
    cv2.imshow('test_image', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    return test_image

recognize_all_faces_in_one_image("test_data/test17.PNG")