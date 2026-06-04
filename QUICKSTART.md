# NeuroScope AI - Deployment Quick Start
# =========================================

## Files in this package
```
neuroscope_deploy/
  src/
    main.py              -- FastAPI application (all endpoints + WebSocket)
    requirements.txt     -- Python dependencies
    ohif_extension.jsx   -- OHIF Viewer React extension
  docker/
    docker-compose.yml   -- 3-tier deployment (gpu / cpu / edge)
    Dockerfile.gpu       -- Tier 1: NVIDIA GPU server
    Dockerfile.cpu       -- Tier 2: CPU workstation
    Dockerfile.edge      -- Tier 3: Offline laptop
    prometheus.yml       -- Monitoring config
```

---

## 1. Local development (Windows, no Docker)

```powershell
cd C:\Users\tejan\OneDrive\Desktop\drive\NeuroScope_AI

# Install deps
pip install fastapi uvicorn python-multipart onnxruntime-gpu opencv-python-headless

# Run API
cd src
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Test
curl http://localhost:8000/health
# -> {"status":"ok","models":7,...}
```

Open API docs: http://localhost:8000/docs

---

## 2. Test with a scan

```python
import requests, base64

# Load a test image
with open('test_brain.jpg', 'rb') as f:
    img_bytes = f.read()

# Sync analysis
resp = requests.post(
    'http://localhost:8000/analyze/sync',
    files={'file': ('scan.jpg', img_bytes, 'image/jpeg')},
    data={
        'cancer_type'   : 'brain',
        'patient_age'   : 45,
        'patient_sex'   : 'M',
        'clinical_notes': 'Headache, visual disturbance',
    }
)
result = resp.json()
print(f"Verdict: {result['verdict']}")
print(f"Cancer prob: {result['cancer_prob']:.1%}")
print(f"Tumor type: {result['tumor_type']}")
```

---

## 3. WebSocket streaming (real-time progress)

```javascript
const scanId = 'test_001';
const ws = new WebSocket(`ws://localhost:8000/ws/analyze/${scanId}`);

ws.onopen = () => {
  ws.send(JSON.stringify({
    cancer_type: 'brain',
    image_b64  : '<base64 image>',
    patient_age: 45,
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(`${msg.stage}: ${msg.status}`);
  // normalization: done
  // triage: done
  // detection: done -> verdict: CANCER_FLAGGED
  // ...
  if (msg.stage === 'result') {
    console.log('Final result:', msg.data);
  }
};
```

---

## 4. Docker deployment

```bash
# Tier 1 -- GPU server (hospital data center)
docker-compose --profile gpu up -d

# Tier 2 -- CPU workstation (clinic)
docker-compose --profile cpu up -d

# Tier 3 -- Edge offline (rural clinic)
docker-compose --profile edge up -d
```

Models volume: mount your models/production folder:
```bash
docker run -v C:/Users/tejan/.../models/production:/app/data/models neuroscope-ai:gpu
```

---

## 5. OHIF Extension

1. Copy `ohif_extension.jsx` to your OHIF extensions folder
2. Register in `app-config.js`:
```javascript
extensions: [
  { id: 'neuroscope-ai', path: './extensions/ohif_extension' }
]
```
3. Set API URL:
```bash
REACT_APP_NEUROSCOPE_API=http://localhost:8000
```
4. The AI panel appears in the right sidebar of the OHIF viewer

---

## 6. API Endpoints

| Method | Endpoint              | Description                        |
|--------|-----------------------|------------------------------------|
| GET    | /health               | Health check                       |
| GET    | /models               | List loaded models                 |
| POST   | /analyze/sync         | Synchronous full pipeline          |
| POST   | /analyze              | Async (returns task_id)            |
| GET    | /status/{task_id}     | Check async task progress          |
| GET    | /report/{scan_id}     | Fetch generated report             |
| POST   | /classify             | Classification only                |
| WS     | /ws/analyze/{scan_id} | Real-time streaming pipeline       |

---

## 7. Next steps (remaining deployment work)

- [ ] Add JWT authentication (`python-jose`)
- [ ] Connect Celery + Redis for production async jobs
- [ ] Add rate limiting (`slowapi`)
- [ ] DICOM endpoint (`/analyze/dicom`) for direct PACS integration
- [ ] Kubernetes Helm chart for multi-node scaling
- [ ] Federated learning nodes (Flower)
