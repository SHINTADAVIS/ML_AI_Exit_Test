# emotion_analysis.py

import cv2
from deepface import DeepFace

# Path to your video
video_path = "data/video.mp4"

# Load face detection model
face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# List to store processed frames
frame_list = []

# Open video
capture = cv2.VideoCapture(video_path)

# Process frames
for i in range(5000):
    ret, frame = capture.read()
    if not ret:
        break

    # Detect faces
    faces = face_model.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)

    for (x, y, w, h) in faces:
        # Analyze emotion
        emotion = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        
        # Draw rectangle and emotion text
        cv2.putText(frame,
                    emotion["dominant_emotion"],
                    (x, y + h),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.9,
                    (255, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        frame_list.append(frame)

# Define output video properties
if frame_list:
    height, width, colors = frame_list[0].shape
    size = (width, height)
    output_path = "Emotions.avi"
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"DIVX"), 10, size)

    for frame in frame_list:
        output.write(frame)

    output.release()
    print(f"Processed video saved as {output_path}")
else:
    print("No frames processed. Check video path or face detection.")
