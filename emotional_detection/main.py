import cv2
import os
import time
import ast
import json
from deepface import DeepFace
from deepface.commons import functions, realtime, distance as dst

print("WHAT's up BRUV")

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def create_output_pipe(pipe_name):
    os.mkfifo(pipe_name)

def write_output(payload):
    global PIPE_NAME
    with open(PIPE_NAME, 'w') as fifo:
        json_data = ast.literal_eval(str(payload))
        print(json_data)
        fifo.write(json.dumps(json_data) + "\n")
        
def analyze_face(face_image):
    # gray_img = functions.preprocess_face(img = face_image, target_size = (48, 48), grayscale = True, enforce_detection = False, detector_backend = 'opencv')
    # emotion_predictions = emotion_model.predict(gray_img)[0,:]
    # print(emotion_predictions)
    # sum_of_predictions = emotion_predictions.sum()
    try:
        return DeepFace.analyze(img_path = face_image, actions = ['emotion'])
    except ValueError:
        # Change this??? Maybe random? Maybe no-op... Make neutral no-op?
        return { 'emotion': {'neutral': 1.0}, 'dominant_emotion': 'neutral' }

# -------------
# SETUP
# -------------
PIPE_NAME = "EMOTIONAL_PIPE"
# create_output_pipe(PIPE_NAME)
tic = time.time()
face_frame_count = 6
freeze = False
freezed_frames = 0
dom_emotion = None

print("DOING THIS")

# Capture video from the primary webcamera
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    if img is None:
        break

    # Convert into grayscale
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    faces = face_cascade.detectMultiScale(img_grayscale, 1.1, 4)

    if faces is None or faces == []:
        face_frame_count = 6


    #  Draw rectangle around the faces
    for (x, y, w, h) in faces:
        if w > 130: # discard small "faces"
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


            if (dom_emotion and freezed_frames < 10):
                freezed_frames += 1
                cv2.putText(img, dom_emotion, (int(x+w/4),int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
                continue


            # Detect the emotions if 5 consecutive frames of a Face
            if face_frame_count == 1:
                raw_img = img.copy()
                custom_face = raw_img[y:y+h, x:x+w]
                emotions = analyze_face(custom_face)
                write_output(emotions)
                dom_emotion = emotions.get('dominant_emotion', 'neutral') if emotions else 'neutral'

                # Start Freeze & face_frame_count
                freeze = True
                face_frame_count = 6
            else:
                freeze = False
                freezed_frames = 0
                face_frame_count -= 1
                dom_emotion = None
                cv2.putText(img, str(face_frame_count), (int(x+w/4),int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)


    # Display the output
    cv2.imshow('img', img)

    # Break on Escape
    key = cv2.waitKey(30) & 0xff
    if key==27:
        break

os.close(w)
cap.release()
