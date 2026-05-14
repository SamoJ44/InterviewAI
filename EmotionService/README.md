# InterviewAI Emotion Service

Separate TensorFlow/Keras service for facial expression prediction. This keeps TensorFlow out of the main MediaPipe backend environment.

## Run

From the project root:

```powershell
python -m venv emotion_venv
.\emotion_venv\Scripts\activate
pip install -r EmotionService\requirements.txt
uvicorn EmotionService.app:app --host 127.0.0.1 --port 8765 --reload
```

The default model path is:

```text
..\Backend\models\best_resnet50v2_finetuned.keras
```

Override it with:

```powershell
$env:EMOTION_MODEL_PATH="C:\path\to\best_resnet50v2_finetuned.keras"
```

## Run The Rest Of InterviewAI

Main backend:

```powershell
cd Backend
..\venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Frontend:

```powershell
cd Frontend
npm run dev
```

## Manual Tests

### A. Test Emotion Service Alone

Open:

```text
http://127.0.0.1:8765/docs
```

Call:

```text
POST /predict-emotion
```

Upload an image or face crop. Confirm the response includes:

- `label`
- `confidence`
- `probabilities`
- `score`
- `available`
- `status`

### B. Test Main Backend Integration

Start both services, then start the frontend.

Start a session and wait for calibration to finish. Confirm the Expression Score card changes from `model_not_loaded` or `service_unavailable` to an actual expression prediction when the emotion service is available.

### C. Test Failure Mode

Stop the EmotionService. The main backend should keep running, `/analyze-frame` should not crash, and the frontend should show expression service unavailable.
