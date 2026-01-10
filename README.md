# Automated Emotion Detection from a video

**Course Exit Test Project – ML & AI Program**

## Emotion Detection from Video using DeepFace and OpenCV

This project demonstrates **automatic emotion recognition from video frames** using **DeepFace** and **OpenCV**.  
The script analyzes each frame of a video, detects faces, predicts the dominant emotion for each detected face, overlays the emotion label, and saves the annotated video.

## Project Overview

The goal of this project is to:  
1. **Extract frames** from a video file    
2. **Detect faces** using a Haar Cascade classifier    
3. **Analyze emotions** using the DeepFace library    
4. **Overlay predictions** (emotion labels) on detected faces    
5. **Save the output video** with all annotations  

## Installation

### 1. Clone the repository
   
git clone https://github.com/YOUR_USERNAME/DeepFace_Emotion.git  
cd DeepFace_Emotion  

### 2. Install Python dependencies  

pip install -r requirements.txt  

### Usage  

#### 1.Provide an MP4 video for testing:  

  Create a folder named data.      
  Place your video inside data/ (e.g., data/video.mp4).      

#### 2.Update the script src/emotion_analysis.py with the video path:  
  video_path = "data/video.mp4"   # Replace with your video path  

#### 3.Run the Python script:  
  python src/emotion_analysis.py  
  
  The script will detect faces, recognize emotions, and annotate frames.  
  An output video will be saved (default: Emotions.avi).  

### Requirements  
Python 3.x  
OpenCV (opencv-python)  
DeepFace (deepface)  
TensorFlow (tensorflow)  
Pandas (pandas)  
NumPy (numpy)  
Pillow (Pillow)  
MTCNN (mtcnn)  
RetinaFace (retina-face)  

All dependencies are included in requirements.txt.  

### Notes
No video file is included in this repository due to size constraints.  
You can use any MP4 video with faces to test the project.  
The original Colab notebook is in notebooks/.  

### Sample Output

Recognized emotions will be annotated on video frames.  

Expected behavior:  
Faces detected and highlighted with rectangles.  
Dominant emotion displayed on each face.  
Output video saved as Emotions.avi.  

