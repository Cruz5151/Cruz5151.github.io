import cv2
import os
import numpy as np


def detect_faces_in_one_image(color_image, scaleFactor=1.1):
    # input: one RGB image
    # output: the face part in the image(gray), the whole image(gray) with rectangle idetifing the face
    # ...
    # one training image is supposed to contain only one face,
    # therefore, the output of the function should only be one face
    
    # convert the original image to gray level image
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # load cascade classifier training file for haarcascade
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    faces = face_detector.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    print('Face found: ', len(faces))

    # draw the rectangle in the detected face region
    for (x,y,w,h) in faces:
        cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 0, 255), 2, cv2.LINE_8, 0)
   
    if (len(faces) == 0):
        return None, None
    else:
        (x,y,w,h) = faces[0]
        return gray_image[y:y+w, x:x+h], faces[0]


def prepare_training_data(data_folder_path):
    # input: the folder path of main file 'training_data'
    # output: two arrays, faces[] and corresponding labels[]
    
    # get the dirctiries (one for each label) in data folder, there are three or more subfiles under "traning_data"
    dirs = os.listdir(data_folder_path)
    
    # list to store all subject faces
    faces = []
    labels = []

    for dir_name in dirs:
        if not dir_name.startswith("c"):
            continue
        
        # extract label number of subject from dir_name(eg. c1, c2), remove 'c' to get the label
        label = int(dir_name.replace("c",""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_image_names = os.listdir(subject_dir_path)

        # go through image, read image, detect face and add face to list of faces
        for image_name in subject_image_names:
            if image_name.startswith("."):
                continue
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)

            cv2.imshow("Training on image ...", image)
            cv2.waitKey(100)
            cv2.destroyAllWindows()

            face, rect = detect_faces_in_one_image(image)

            # ignore faces that are not detected
            if face is not None:
                faces.append(face)
                labels.append(label)            

    return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data("resized_training_data")
print("Data prepared")

print("Total faces: ", len(faces))
print("total labels: ", len(labels))

# create the collection of label, e.g. labels[3] represents 'Siliang' (Myself)
label_collection = ["","Boris","Otis","Siliang"]

print("start training the model")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
face_recognizer.save('trained_recognizer.yml')
print("model training finish")