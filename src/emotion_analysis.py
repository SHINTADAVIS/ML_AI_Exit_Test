# emotion_analysis.py

import os
import cv2
from deepface import DeepFace
from google.colab.patches import cv2_imshow  # Optional if running in Colab

# --- User must update this path ---
video_path = "data/video.mp4"

# Check if video exists
if not os.path.isfile(video_path):
    print(f"Error: Video file not found at '{video_path}'")
    print("Please provide your own MP4 video and update the 'video_path' variable.")
    exit(1)  # Stop execution

# Load face detection model
face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Prepare to store frames
frame_list = []

# Open video
capture = cv2.VideoCapture(video_path)
if not capture.isOpened():
    print(f"Error: Cannot open video '{video_path}'")
    exit(1)

# Process video frames
for i in range(5000):
    ret, frame = capture.read()
    if not ret:
        break  # Stop if no more frames

    faces = face_model.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)

    for (x, y, w, h) in faces:
        try:
            # Analyze emotions
            emotion = DeepFace.analyze(frame, actions=["emotion"])
            # Put emotion text on frame
            cv2.putText(frame,
                        str(emotion["dominant_emotion"]),
                        (x, y + h),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.9,
                        (255, 255, 0),
                        2)
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        except Exception as e:
            print(f"Warning: Could not analyze emotion for a face. Error: {e}")

    frame_list.append(frame)

# Prepare output video
if frame_list:
    height, width, _ = frame_list[0].shape
    size = (width, height)
    output_path = "Emotions.avi"
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"DIVX"), 10, size)

    for frame in frame_list:
        output.write(frame)

    output.release()
    print(f"Output video saved as '{output_path}'")
else:
    print("No frames were processed. Check your video file.")

