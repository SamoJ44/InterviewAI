# Environment Notes

This project currently has two different runtime needs:

- MediaPipe webcam backend: `Backend/main.py`, `Backend/api.py`, scoring pipeline.
- TensorFlow emotion testing: `Backend/real_time_test.py`.

Use separate virtual environments if TensorFlow requires protobuf 6 or newer.
The MediaPipe backend should use:

```text
mediapipe==0.10.20
protobuf>=4.25.3,<5
```

`protobuf` is pinned below 5 because newer protobuf releases can break
MediaPipe FaceMesh/Pose initialization with descriptor API errors.

## Dependency Checks

Run these inside the environment you want to validate:

```powershell
python -m pip show mediapipe protobuf
python -m pip check
python -c "import google.protobuf; import mediapipe as mp; print(google.protobuf.__version__, mp.__version__)"
```

## Repair MediaPipe Backend Environment

For the webcam/API backend environment, remove TensorFlow packages first if
they force an incompatible protobuf version:

```powershell
.\venv\Scripts\python.exe -m pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu keras mediapipe protobuf
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then test:

```powershell
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe Backend\main.py
```

## TensorFlow Emotion Test Environment

If you need `Backend/real_time_test.py`, create a separate virtual environment
for TensorFlow so it can use the protobuf version required by TensorFlow
without breaking MediaPipe.
