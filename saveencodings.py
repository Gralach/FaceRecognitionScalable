import math
from sklearn import neighbors
import os
import os.path
import pickle
import cv2
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import time
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


DIRECTORY = "known_image/"
ENCODING_PATH = "encodings/"

for class_dir in os.listdir(DIRECTORY):
        if not os.path.isdir(os.path.join(DIRECTORY, class_dir)):
            continue
        elif os.path.exists(ENCODING_PATH+class_dir):
            continue
        else:
            for img_path in image_files_in_folder(os.path.join(DIRECTORY, class_dir)):
                print(img_path)
                frame = cv2.imread(img_path)
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                face_bounding_boxes = face_recognition.face_locations(rgb_small_frame)
                print(len(face_bounding_boxes))
                for (top, right, bottom, left) in face_bounding_boxes:
                  cv2.rectangle(small_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.imshow('Video', small_frame)
                cv2.waitKey()
                cv2.destroyAllWindows()
                if len(face_bounding_boxes)!=1:
                   print("Photo not suitable")
                else:
                   encoding = face_recognition.face_encodings(small_frame, known_face_locations=face_bounding_boxes)[0]
                   print(len(encoding))
                   #encodings.append(encoding)
                   location = ENCODING_PATH + class_dir
                   if os.path.exists(location):
                      print("Encoding skipped")
                   elif not os.path.isdir(location):
                      os.mkdir(location)
                      simpan = open(str(location)+'/'+class_dir+ '.txt','w')
                      np.savetxt(simpan,encoding)
                      simpan.close()
                      print("Encoding Saved")
