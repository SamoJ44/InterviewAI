# InterviewAI  
### Real-Time Computer Vision System for Interview Behavior Analysis

InterviewAI is a real-time computer vision system that analyzes a candidate’s non-verbal behavior during mock interviews using webcam video.

The system evaluates:

- Eye contact
- Posture
- Body stability
- Self-touch gestures
- Facial expression / emotion

It combines landmark-based computer vision, real-time behavioral scoring, and deep learning-based facial emotion recognition to provide live interview feedback through an interactive web dashboard.

---

## 1. Project Motivation

In interviews, non-verbal behavior can strongly affect perceived confidence and communication quality. Many candidates struggle to objectively assess their posture, eye contact, nervous gestures, and facial expressions while practicing.

InterviewAI addresses this problem by providing a real-time automated feedback system that analyzes visible behavioral cues from video frames and converts them into interpretable scores.

---

## 2. System Overview

The project is organized into three main components:

```text
IAI/
│
├── Backend/          # Main FastAPI backend and computer vision pipeline
├── EmotionService/   # Separate facial emotion recognition service
├── Frontend/         # React/Vite live dashboard
├── requirements.txt  # Main backend Python dependencies
└── README.md
```

### Processing Flow

1. The frontend captures webcam frames.
2. Frames are sent to the FastAPI backend.
3. The backend extracts visual cues using face, pose, and hand landmarks.
4. The backend calls the emotion recognition service for facial expression prediction.
5. Scores, flags, events, and emotion outputs are returned to the frontend.
6. The dashboard displays live metrics and a final interview summary.

---

## 3. Main Features

### Real-Time Behavioral Analysis

- **Eye Contact Score**  
  Estimates whether the user is visually oriented toward the camera.

- **Posture Score**  
  Evaluates upper-body alignment and posture consistency.

- **Stability Score**  
  Measures excessive body motion or restlessness over time.

- **Self-Touch Detection**  
  Detects hand contact with body/head regions such as face, hair, neck, and chest.

- **Facial Emotion Recognition**  
  Predicts facial expression probabilities using a trained deep learning model.

- **Final Interview Score**  
  Aggregates multiple behavioral scores into one live performance estimate.

### Frontend Dashboard

- Live webcam session
- Real-time score cards
- Emotion prediction card
- Detection status badges
- Event history panel
- Final session summary

---

## 4. Technical Approach

### 4.1 Computer Vision Pipeline

The main backend performs frame-by-frame behavioral analysis using visual landmarks:

- Face landmarks for eye-contact related logic
- Pose landmarks for posture and stability estimation
- Hand landmarks for self-touch detection
- Temporal smoothing to avoid unstable frame-by-frame outputs
- Event tracking for behaviors such as prolonged gaze-away or self-touch activity

### 4.2 Emotion Recognition

Facial emotion recognition is handled by a separate service using a fine-tuned deep learning model.

- Backbone architecture: **ResNet50V2**
- Pretrained using ImageNet weights
- Fine-tuned for facial emotion classification
- Output includes:
  - Predicted emotion label
  - Confidence score
  - Full probability distribution over emotions

---

## 5. Dataset

The facial emotion recognition model was trained using the public:

**FER2013-Enhanced dataset**

The model predicts the following emotion classes:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The dataset was split into training, validation, and testing subsets during model development.

---

## 6. Repository Structure

```text
Backend/
├── api.py
├── config.py
├── detectors.py
├── emotion_detector.py
├── main.py
├── pipeline.py
├── real_time_test.py
├── recommendation_service.py
├── scoring_diagnostics.py
├── touch_logic.py
├── utils.py
└── visualization.py

EmotionService/
├── app.py
├── requirements.txt
└── README.md

Frontend/
├── src/
├── package.json
├── package-lock.json
├── tsconfig.json
└── vite.config.ts
```

---

## 7. Installation

### 7.1 Clone the Repository

```bash
git clone https://github.com/SamoJ44/InterviewAI.git
cd InterviewAI
```

---

### 7.2 Create and Activate a Python Virtual Environment

```bash
python -m venv .venv
```

#### Windows PowerShell

```bash
.venv\Scripts\Activate.ps1
```

---

### 7.3 Install Main Backend Dependencies

```bash
pip install -r requirements.txt
```

---

### 7.4 Install Emotion Service Dependencies

```bash
pip install -r EmotionService/requirements.txt
```

---

### 7.5 Install Frontend Dependencies

```bash
cd Frontend
npm install
cd ..
```

---

## 8. Model Weights

The trained emotion recognition model is not included directly in the GitHub repository because the file is too large for standard GitHub storage.

Expected local model path:

```text
Backend/models/best_resnet50v2_finetuned.keras
```

To enable facial emotion recognition:

1. Create the folder:

```text
Backend/models/
```

2. Place the trained model file inside it with the exact name:

```text
best_resnet50v2_finetuned.keras
```

3. Add the final public model download link here before submission:

```text
Model weights download link: https://drive.google.com/file/d/1ua5JXuovZg5WAb5YQH5wKqYaeSTvgJlL/view?usp=sharing
```

---

## 9. How to Run the Full System

The system requires three services to run:

1. Main backend
2. Emotion recognition service
3. Frontend dashboard

---

### 9.1 Run the Main Backend

From the project root folder:

```bash
python -m uvicorn Backend.api:app --host 127.0.0.1 --port 8000 --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

---

### 9.2 Run the Emotion Recognition Service

Open a second terminal and run:

```bash
cd EmotionService
python -m uvicorn app:app --host 127.0.0.1 --port 8765 --reload
```

The emotion service will run at:

```text
http://127.0.0.1:8765
```

---

### 9.3 Run the Frontend

Open a third terminal and run:

```bash
cd Frontend
npm run dev
```

The frontend will launch locally, usually at:

```text
http://localhost:5173
```

---

## 10. Backend API

### Health Check

```http
GET /health
```

Used to verify that the backend is running.

---

### Frame Analysis

```http
POST /analyze-frame
```

Receives a webcam frame and returns:

- Session ID
- Behavioral scores
- Detection flags
- Emotion prediction
- Frame metadata
- Events and alerts

Example response structure:

```json
{
  "session_id": "example-session",
  "scores": {
    "eye_contact": 82.5,
    "posture": 76.0,
    "stability": 88.4,
    "self_touch": 91.2,
    "expression": 70.0,
    "final": 81.6
  },
  "flags": {
    "face_detected": true,
    "pose_detected": true,
    "hands_detected": false,
    "self_touch_active": false
  },
  "emotion": {
    "label": "happy",
    "confidence": 0.87
  }
}
```

---

## 11. Important Design Choices

Several implementation choices were made to improve real-time behavior and output quality:

- Modular architecture with separated backend, frontend, and emotion service
- Landmark-based interpretable scoring instead of black-box behavior scoring
- Temporal smoothing for live score stability
- Time-normalized movement measurement for stability analysis
- Refined self-touch logic to reduce false positives from hovering hands
- Separate emotion inference service for cleaner system modularity
- Fine-tuned CNN model for facial expression prediction

---

## 12. Limitations

- Performance depends on webcam quality, lighting, and face visibility.
- Eye-contact estimation is approximate and camera-relative.
- Self-touch detection remains challenging under occlusion or ambiguous depth.
- Facial emotion prediction estimates expression, not a person’s actual mental state.
- Results may vary across users, poses, and environmental conditions.

---

## 13. Future Improvements

- Train a dedicated interview-specific emotion model
- Improve depth-aware self-touch estimation
- Add richer final reports after each interview session
- Add historical progress tracking across sessions
- Perform more systematic quantitative evaluation of behavioral scoring modules
- Add speach recognition & Analysis

---

## 14. Authors

**Computer Vision Final Project**  
Samir Jabara
John El Hachem
Charbel Mouawad
Saint Joseph University — ESIB  
Spring 2026

