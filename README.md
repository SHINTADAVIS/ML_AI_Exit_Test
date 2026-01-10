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



   
        

#### Requirements

Install the following dependencies: 

    !pip install deepface opencv-python
    
DeepFace automatically installs required backend frameworks (TensorFlow, Keras, etc.) if not already present.  

#### Notes and Recommendations

If the video has more than 5000 frames, adjust the loop limit:  

    for i in range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):
    
To speed up processing, consider analyzing every Nth frame instead of all frames.    
GPU acceleration is recommended for faster DeepFace inference (available in Google Colab)  

