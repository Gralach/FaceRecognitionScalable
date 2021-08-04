import os
import os.path
import face_recognition
import numpy as np
import cv2
import argparse
from face_recognition.face_recognition_cli import image_files_in_folder
from tqdm import tqdm
from collections import defaultdict
from eye_status import *


# Options
# -------
parser = argparse.ArgumentParser(description='Face Recogntion')
parser.add_argument('--add_encodings', action='store_true')
parser.add_argument('--update_encodings', nargs = 2, default = ("../FaceRec_verDlib/known_image/Aaron_Eckhart," "A" ))
parser.add_argument('--load_encodings', action='store_true')
parser.add_argument('--video_cam', action='store_true')
opt = parser.parse_args()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
DIRECTORY = "known_image/"
ENCODING_PATH = "encodings/"
known_face_encodings = []
known_face_names = []

def init():
    open_eye_cascPath = 'cascades/haarcascade_eye_tree_eyeglasses.xml'
    left_eye_cascPath = 'cascades/haarcascade_lefteye_2splits.xml'
    right_eye_cascPath ='cascades/haarcascade_righteye_2splits.xml'

    open_eyes_detector = cv2.CascadeClassifier(open_eye_cascPath)
    left_eye_detector = cv2.CascadeClassifier(left_eye_cascPath)
    right_eye_detector = cv2.CascadeClassifier(right_eye_cascPath)

    model = load_model()

    return (model, open_eyes_detector, left_eye_detector, right_eye_detector)

def load():
    known_face_encodings = []
    known_face_names = []

    for class_dir in os.listdir(ENCODING_PATH):
        #encodings
        info = ENCODING_PATH + class_dir + "/" + class_dir + ".txt"
        original_encoding = np.loadtxt(info)
        known_face_encodings.append(original_encoding)
        known_face_names.append(class_dir)
    return known_face_names, known_face_encodings

def isBlinking(history, maxFrames):
    """ from AI3000 """
    for i in range(maxFrames):
        pattern = '1' + '0'*(i+1) + '1'
        if pattern in history:
            return True
    return False

if opt.update_encodings:
    update_dir = opt.update_encodings[0]
    name = opt.update_encodings[1]
    for img_path in image_files_in_folder(update_dir):
                frame = cv2.imread(img_path)
            
                if frame.shape[0] > 1000 and frame.shape[1] > 1000:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                else:
                    small_frame = frame
                rgb_small_frame = small_frame[:, :, ::-1]
                face_bounding_boxes = face_recognition.face_locations(rgb_small_frame)
                print(len(face_bounding_boxes))
                if len(face_bounding_boxes)!=1:
                   print("Photo not suitable")
                else:
                   encoding = face_recognition.face_encodings(small_frame, known_face_locations=face_bounding_boxes)[0]
                   print(len(encoding))

                   location = ENCODING_PATH + name
                   if not os.path.isdir(location):
                      os.mkdir(location)
                      simpan = open(str(location)+'/'+name+ '.txt','w')
                      np.savetxt(simpan,encoding)
                      simpan.close()
                      print("Encoding Saved")
    

if opt.add_encodings:
    for class_dir in os.listdir(DIRECTORY):
        if not os.path.isdir(os.path.join(DIRECTORY, class_dir)):
            continue
        elif os.path.exists(ENCODING_PATH+class_dir):
            continue
        else:
            for img_path in image_files_in_folder(os.path.join(DIRECTORY, class_dir)):
                frame = cv2.imread(img_path)
            
                if frame.shape[0] > 1000 and frame.shape[1] > 1000:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                else:
                    small_frame = frame
                rgb_small_frame = small_frame[:, :, ::-1]
                face_bounding_boxes = face_recognition.face_locations(rgb_small_frame)
                print(len(face_bounding_boxes))
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

if opt.load_encodings:
    known_face_names, known_face_encodings = load()

if opt.video_cam:    
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    video_capture = cv2.VideoCapture(0)
    counter = 0
    if len(known_face_encodings) == 0:
        known_face_names, known_face_encodings = load()
    (model, open_eyes_detector, left_eye_detector, right_eye_detector) = init()

    eyes_detected = defaultdict(str)
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
        face_locations = face_recognition.face_locations(rgb_small_frame)
        if process_this_frame:
            process_this_frame = not process_this_frame
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                 # See if the face is a match for the known face(s)
                 matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance = 0.35)
                 name = "Unknown"
                 score = "Unknown"
                 face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                 best_match_index = np.argmin(face_distances)
                 if matches[best_match_index]:
                     score = str(round(face_distances[best_match_index],3))
                     name = known_face_names[best_match_index]
                 face_names.append(name)

        if len(face_locations) != 0:
            
            face = small_frame[face_locations[0][0]:face_locations[0][2],face_locations[0][1]:face_locations[0][3]]
            gray_face = gray[face_locations[0][0]:face_locations[0][2],face_locations[0][1]:face_locations[0][3]]
            eyes = []
            
            # Eyes detection
            # check first if eyes are open (with glasses taking into account)
            open_eyes_glasses = open_eyes_detector.detectMultiScale(
                gray_face,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
            # if open_eyes_glasses detect eyes then they are open 
            if len(open_eyes_glasses) == 2:
                eyes_detected[name]+='1'
                for (ex,ey,ew,eh) in open_eyes_glasses:
                    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                
            # otherwise try detecting eyes using left and right_eye_detector
            # which can detect open and closed eyes                
            else:
                # separate the face into left and right sides
                left_face = rgb_small_frame[face_locations[0][0]:face_locations[0][2],int(face_locations[0][3]/2):face_locations[0][3]]
                left_face_gray = gray[face_locations[0][0]:face_locations[0][2],int(face_locations[0][3]/2)::face_locations[0][3]]
                
                right_face = rgb_small_frame[face_locations[0][0]:face_locations[0][2],face_locations[0][1]:int(face_locations[0][3]/2)]
                right_face_gray = gray[face_locations[0][0]:face_locations[0][2],face_locations[0][1]:int(face_locations[0][3]/2)]
                # Detect the left eye
                left_eye = left_eye_detector.detectMultiScale(
                    left_face_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )
                # Detect the right eye
                right_eye = right_eye_detector.detectMultiScale(
                    right_face_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )
    
                eye_status = '1' # we suppose the eyes are open
    
                # For each eye check wether the eye is closed.
                # If one is closed we conclude the eyes are closed
                for (ex,ey,ew,eh) in right_eye:
                    color = (0,255,0)
                    pred = predict(right_face[ey:ey+eh,ex:ex+ew],model)
                    if pred == 'closed':
                        eye_status='0'
                        color = (0,0,255)
                    cv2.rectangle(right_face,(ex,ey),(ex+ew,ey+eh),color,2)
                for (ex,ey,ew,eh) in left_eye:
                    color = (0,255,0)
                    pred = predict(left_face[ey:ey+eh,ex:ex+ew],model)
                    if pred == 'closed':
                        eye_status='0'
                        color = (0,0,255)
                    cv2.rectangle(left_face,(ex,ey),(ex+ew,ey+eh),color,2)
                eyes_detected[name] += eye_status
        
        counter = counter+1
        if counter == 30:
            counter = 0        
            process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            if isBlinking(eyes_detected[name],3):
                # Display name
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
                cv2.putText(frame, score, (left + 6, top - 6), font, 0.8, (255, 255, 255), 1)
            else:
                print(name)
                fake = "Fake"
                # Display name
                cv2.putText(frame, fake, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
                cv2.putText(frame, fake, (left + 6, top - 6), font, 0.8, (255, 255, 255), 1)
                

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()