# InterviewAI Frontend

Premium React + Vite + TypeScript live interview session UI for the FastAPI backend.

## Backend

```powershell
cd Backend
..\venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

## Frontend

```powershell
cd Frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Environment

The frontend reads:

```text
VITE_BACKEND_URL=http://127.0.0.1:8000
```

If it is missing, the app defaults to `http://127.0.0.1:8000`.

## Manual Test

1. Start backend.
2. Start frontend.
3. Open frontend.
4. Toggle dark/light mode.
5. Click Start Session.
6. Allow webcam.
7. Confirm webcam preview appears mirrored.
8. Confirm live scores update.
9. Confirm detection flags update.
10. Click Pause.
11. Confirm analysis stops but UI remains.
12. Click Resume.
13. Confirm analysis resumes.
14. Click End.
15. Confirm camera stops and final summary appears.
