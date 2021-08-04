import os
import os.path
import face_recognition
import numpy as np
import cv2

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

ENCODING_PATH = "encodings/"

known_face_encodings = []
known_face_names = []

for class_dir in os.listdir(ENCODING_PATH):
    #encodings
    info = ENCODING_PATH + class_dir + "/" + class_dir + ".txt"
    original_encoding = np.loadtxt(info)
    known_face_encodings.append(original_encoding)
    known_face_names.append(class_dir)
    
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

video_capture = cv2.VideoCapture(0)
counter = 0


while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    if process_this_frame:
        process_this_frame = not process_this_frame
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
             # See if the face is a match for the known face(s)
             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
             name = "Unknown"
             score = "Unknown"
             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
             best_match_index = np.argmin(face_distances)
             if matches[best_match_index]:
                 score = str(round(face_distances[best_match_index],3))
                 print(score)
                 name = known_face_names[best_match_index]
             face_names.append(name)
             
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
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        cv2.putText(frame, score, (left + 6, top - 6), font, 0.8, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

