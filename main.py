# pip install cmake // cmake is dependent library for face_recognition
# pip install dlib // dlib is dependent library for face_recognition
# pip install face_recognition
# pip install opencv-python

import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)
# Accessing the webcam to read video to cv2. 0 indicates first webcam device

# Load Known Faces
harry_image = face_recognition.load_image_file("faces/harry.jpg")
harry_encoding = face_recognition.face_encodings(harry_image)[0]
# Encodings will convert the image to mathematical values. So it will be easier to compare to other images.
# [0] indicates only returning the first face in the form of a list.
rohan_image = face_recognition.load_image_file("faces/rohan.jpg")
rohan_encoding = face_recognition.face_encodings(rohan_image)[0]

# Storing names of the encodings
known_face_encodings = [harry_encoding, rohan_encoding]
# array to store the encodings
known_face_names = ["Harry", "Rohan"]
# array to store names for these encodings

# list of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# Get the current date and time at the time of locking the attendance
now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    # Convert the colour of the frame
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces on the webcam and convert it into encodings
    face_locations = face_recognition.face_locations()

    for face_encoding in face_encodings:
        # Compares the recorded faces in webcam to the stored encodings and returns true or false
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        # Checks the similarity between the faces using euclidean distance
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if (matches[best_match_index]):
            # To get the name of the face that has matched
            name = known_face_names[best_match_index]

    #     Display the text if a person is present
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

    #         Remove the name of the student from list after being marked present and write the names to csv files
            if name in students:
                students.remove(name)
                current_time = now.strftime(%H-%M-%S)
                lnwriter = writerow([name, current_time])

    # Display the frame
    cv2.imshow("Attendance", frame)
    # To close the window by pressing q
    if cv2.waitkey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()