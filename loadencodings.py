import os
import os.path
import face_recognition
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

DIRECTORY = "known_image/"
ENCODING_PATH = "encodings/"
TEST = "knn_examples/test"
accuracy = 0
counter = 0
unknownpassed = 0

#file
known_encodings = []
#encodings
known_face_encodings = []
known_face_names = []

for class_dir in os.listdir(ENCODING_PATH):
    #encodings
    info = ENCODING_PATH + class_dir + "/" + class_dir + ".txt"
    known_encodings.append(class_dir)
    #known_encodings.append(info)
    original_encoding = np.loadtxt(info)
    known_face_encodings.append(original_encoding)
    known_face_names.append(class_dir)
    
known_encodings = set(known_encodings)

for image_file in os.listdir("knn_examples/test"):
    img_path = os.path.join("knn_examples/test", image_file)
    truename = image_file.split("_")
    if len(truename) == 3:
       truename = truename[0]+"_"+truename[1]
    else:
       truename=truename[0]
    if not truename in known_encodings:
       continue
    unknown_image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    name = "Unknown"
    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance = 0.52)
        if matches[best_match_index]:
           name = known_face_names[best_match_index]
        if name == "Unknown":
           unknownpassed += 1
           continue
        elif name == truename:
           accuracy = accuracy + 1
        print("Number of Image Correctly Recognized = " + str(accuracy))
        counter += 1
        score = accuracy/counter
        print("Accuracy = " + str(score))
        print("Unknown/Bad Images = " + str(unknownpassed))
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        score = str(face_distances[best_match_index])
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance = 0.52)
        if matches[best_match_index]:
           name = known_face_names[best_match_index]
        score = score.encode("UTF-8")
        text_width, text_height = draw.textsize(name)
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        draw.text((left+ 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
        draw.text((left + 6, top - text_height + 12), score, fill=(255, 255, 255, 255))
    del draw
    pil_image.show()
    time.sleep(2)
    """
    
