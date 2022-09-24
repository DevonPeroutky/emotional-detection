import cv2
import os
import time
import ast
import json
import imutils
from deepface import DeepFace
from deepface.commons import functions, realtime, distance as dst

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def create_output_pipe(pipe_name):
    os.mkfifo(pipe_name)

def write_output(payload):
    global PIPE_NAME
    with open(PIPE_NAME, 'w') as fifo:
        # Convert double quotes to single quotes.
        json_data = ast.literal_eval(str(payload))
        fifo.write(json.dumps(json_data) + '\n')
        
def analyze_face(face_image):
    try:
        return DeepFace.analyze(img_path = face_image, actions = ['emotion'])
    except ValueError:
        # Change this??? Maybe random? Maybe no-op... Make neutral no-op?
        print("Error analzing face")
        return {}

# -------------
# SETUP
# -------------
PIPE_NAME = "/tmp/EMOTIONAL_PIPE"
tic = time.time()
face_frame_count_threshold = 3
face_frame_count = face_frame_count_threshold

# Capture video from the primary webcamera
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    if img is None:
        break

    # Resize image to full screen
    img = imutils.resize(img, width=1120)

    # Convert into grayscale
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    faces = face_cascade.detectMultiScale(img_grayscale, 1.1, 4)

    # Discard "small" (erroneous) faces
    filtered_faces = [(x, y, w, h) for (x, y, w, h) in faces if w > 130]

    # If no faces were detected, simply loop
    if filtered_faces is None or filtered_faces == []:
        face_frame_count = face_frame_count_threshold
        continue

    face_frame_count -= 1
    emotion_hotspots = []

    for (x, y, w, h) in filtered_faces:

        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect the emotions if enough consecutive frames of a Face
        if face_frame_count == 1:
            raw_img = img.copy()
            custom_face = raw_img[y:y+h, x:x+w]
            emotion = analyze_face(custom_face)
            if emotion:
                emotion['region']['x'] = x + w/2
                emotion['region']['y'] = y + h/2
                emotion_hotspots.append(emotion)
            cv2.putText(img, "OUTPUT", (int(x+w/4),int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)

            # Reset face_frame_count
            face_frame_count = face_frame_count_threshold
        else:
            # cv2.putText(img, str(x+w/4), (int(x+w/4),int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
            cv2.putText(img, str(w), (int(x+w/2),int(y+h/1)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)

    e = [e for e in emotion_hotspots if e]
    if e:
        write_output(e)

    # Display the output
    cv2.imshow('img', img)

    # Break on Escape
    key = cv2.waitKey(30) & 0xff
    if key==27:
        break

os.close(w)
cap.release()
