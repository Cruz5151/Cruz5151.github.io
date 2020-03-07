import cv2
import os

# this python file is use for processing the raw training data

dirs = os.listdir("training_data")

for dir_name in dirs:
    subject_dir_path = "training_data" + "/" + dir_name
    subject_image_names = os.listdir(subject_dir_path)
    
    i = 1
    for image_name in subject_image_names:
        image_path = subject_dir_path + "/" + image_name
        image = cv2.imread(image_path)
        
        image = cv2.resize(image, (500,700))
        filename = "resized_training_data"+"/" + dir_name + "/" + str(i) + ".jpg"
        cv2.imwrite(filename, image)
        i = i + 1

print("rescale finish")
