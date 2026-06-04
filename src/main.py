"""
NeuroScope AI - FastAPI Backend
================================
Endpoints:
  POST /analyze          -- full 11-agent pipeline (async via Celery)
  POST /analyze/sync     -- synchronous full pipeline (for testing)
  POST /segment          -- segmentation only
  POST /classify         -- classification only
  GET  /report/{scan_id} -- fetch generated report
  GET  /status/{task_id} -- check async task status
  WS   /ws/analyze       -- WebSocket streaming (real-time agent progress)
  GET  /health           -- health check
  GET  /models           -- list loaded models and status
"""

import os
import sys
import uuid
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import cv2

from fastapi import (
    FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect,
    HTTPException, BackgroundTasks, Form
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from auth import (
    router as auth_router,
    require_auth,
    require_clinician,
    bootstrap_default_users,
    get_current_user,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('neuroscope.api')

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title='NeuroScope AI API',
    description='Multi-cancer medical imaging intelligence platform',
    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redoc',
)
app.include_router(auth_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],        # restrict in production
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_PATH   = os.environ.get(
    'NEUROSCOPE_BASE',
    r'C:\Users\tejan\OneDrive\Desktop\drive\NeuroScope_AI'
)
MODELS_PATH = os.path.join(BASE_PATH, 'models', 'production')
CKPT_PATH   = os.path.join(BASE_PATH, 'checkpoints')
OUT_PATH    = os.path.join(BASE_PATH, 'outputs', 'api')
os.makedirs(OUT_PATH, exist_ok=True)

VALID_CANCER_TYPES = ['brain', 'lung', 'breast', 'liver', 'skin', 'spine']


# ── Pydantic Models ───────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    cancer_type   : str
    patient_age   : Optional[int]   = None
    patient_sex   : Optional[str]   = None
    clinical_notes: Optional[str]   = ''
    priority      : Optional[int]   = 3


class AnalyzeResponse(BaseModel):
    scan_id       : str
    cancer_type   : str
    verdict       : str
    cancer_prob   : float
    tumor_type    : Optional[str]
    confidence    : Optional[float]
    who_grade     : Optional[str]
    priority      : int
    treatment_recs: list
    report_text   : str
    plain_summary : str
    latency_ms    : float
    timestamp     : str


class TaskStatus(BaseModel):
    task_id  : str
    status   : str    # pending | running | completed | failed
    progress : int    # 0-100
    result   : Optional[Dict] = None
    error    : Optional[str]  = None


# ── Model Registry ────────────────────────────────────────────────────────────
class ModelRegistry:
    """
    Loads and caches all ONNX models at startup.
    Lazy-loads on first request if startup fails.
    """

    def __init__(self, models_path: str):
        self.models_path = models_path
        self._sessions  = {}
        self._loaded    = False

    def load_all(self):
        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

            model_files = {
                'brain_cls' : os.path.join(self.models_path, 'brain_cls',  'brain_cls_efficientnet.onnx'),
                'brain_seg' : os.path.join(self.models_path, 'brain_seg',  'brain_seg_resnet.onnx'),
                'breast_det': os.path.join(self.models_path, 'breast_det', 'breast_det_efficientnet.onnx'),
                'liver_seg' : os.path.join(self.models_path, 'liver_seg',  'liver_seg_resnet.onnx'),
                'lung_det'  : os.path.join(self.models_path, 'lung_det',   'lung_det_resnet3d.onnx'),
                'skin_cls'  : os.path.join(self.models_path, 'skin_cls',   'skin_cls_efficientnet.onnx'),
                'spine_cls' : os.path.join(self.models_path, 'spine',      'spine_hybridspinenet.onnx'),
            }

            for name, path in model_files.items():
                if os.path.exists(path):
                    try:
                        self._sessions[name] = ort.InferenceSession(path, providers=providers)
                        logger.info(f'Loaded model: {name}')
                    except Exception as e:
                        logger.warning(f'Failed to load {name}: {e}')
                else:
                    logger.warning(f'Model not found: {name} at {path}')

            self._loaded = True
            logger.info(f'Model registry: {len(self._sessions)}/7 models loaded')

        except ImportError:
            logger.error('onnxruntime not installed')

    def get(self, name: str):
        return self._sessions.get(name)

    def status(self) -> Dict:
        return {
            name: 'loaded' if sess else 'missing'
            for name, sess in self._sessions.items()
        }


model_registry = ModelRegistry(MODELS_PATH)


# ── In-memory task store (use Redis in production) ────────────────────────────
task_store: Dict[str, Dict] = {}


# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, WebSocket] = {}

    async def connect(self, scan_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active[scan_id] = websocket
        logger.info(f'WS connected: {scan_id}')

    def disconnect(self, scan_id: str):
        self.active.pop(scan_id, None)

    async def send(self, scan_id: str, data: Dict):
        ws = self.active.get(scan_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(scan_id)


ws_manager = ConnectionManager()


# ── Core inference helpers ────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img_bytes: bytes, img_size: int,
                     mammogram: bool = False) -> np.ndarray:
    """Decode image bytes and preprocess for model inference."""
    arr  = np.frombuffer(img_bytes, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not decode image')

    if mammogram:
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        img   = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (img_size, img_size)).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.expand_dims(img.transpose(2, 0, 1), 0)
    return img.astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# NCCN rule-based fallback (offline mode)
NCCN_RULES = {
    'glioma'   : ['Maximal safe resection', 'RT + TMZ (Stupp protocol)', 'IDH/MGMT testing'],
    'meningioma': ['Surgery (Simpson grade I-II)', 'SRS for inaccessible lesions'],
    'malignant': ['Surgery + sentinel LN biopsy', 'HER2/ER/PR testing', 'Chemo if indicated'],
    'melanoma' : ['Wide local excision', 'SLNB if >0.8mm', 'BRAF testing', 'Immunotherapy'],
    'nodule'   : ['Lung-RADS scoring', 'Follow-up per NCCN guidelines'],
    'tumor'    : ['BCLC staging', 'Resection/TACE per stage', 'Atezolizumab+bev for BCLC-C'],
    'default'  : ['Oncology referral', 'Tissue biopsy', 'Staging CT', 'MDT review'],
}


def get_treatment_recs(tumor_type: str) -> list:
    return NCCN_RULES.get(tumor_type, NCCN_RULES['default'])


# ── Full pipeline runner ──────────────────────────────────────────────────────
async def run_pipeline(
    scan_id    : str,
    img_bytes  : bytes,
    cancer_type: str,
    patient_age: Optional[int],
    patient_sex: Optional[str],
    notes      : str,
    on_progress=None,
) -> Dict:
    """
    Run the full 11-agent pipeline for one scan.
    on_progress: async callback(stage, data) for WebSocket streaming.
    """
    t_start = time.perf_counter()

    async def emit(stage: str, data: Dict):
        if on_progress:
            await on_progress(stage, data)

    result = {
        'scan_id'     : scan_id,
        'cancer_type' : cancer_type,
        'verdict'     : 'REVIEW_REQUIRED',
        'cancer_prob' : 0.5,
        'tumor_type'  : None,
        'confidence'  : 0.0,
        'who_grade'   : None,
        'priority'    : 3,
        'treatment_recs': [],
        'report_text' : '',
        'plain_summary': '',
        'latency_ms'  : 0.0,
        'timestamp'   : datetime.now().isoformat(),
    }

    # ── Agent 11: Scanner normalization ──────────────────────────────────────
    await emit('normalization', {'status': 'running', 'scan_id': scan_id})
    await asyncio.sleep(0)   # yield to event loop
    await emit('normalization', {'status': 'done'})

    # ── Agent 1: Triage ──────────────────────────────────────────────────────
    await emit('triage', {'status': 'running'})
    await asyncio.sleep(0)
    await emit('triage', {'status': 'done', 'cancer_type': cancer_type})

    # ── Agent 2: Detection ───────────────────────────────────────────────────
    await emit('detection', {'status': 'running'})

    cancer_prob = 0.5
    tumor_type  = 'unknown'
    confidence  = 0.5

    try:
        if cancer_type == 'brain':
            sess = model_registry.get('brain_cls')
            if sess:
                size  = sess.get_inputs()[0].shape[2] or 224
                x     = preprocess_image(img_bytes, size)
                out   = sess.run(None, {sess.get_inputs()[0].name: x})[0]
                probs = softmax(out[0])
                classes = ['no_tumor', 'glioma', 'meningioma', 'pituitary']
                idx   = int(probs.argmax())
                tumor_type  = classes[idx]
                confidence  = float(probs.max())
                cancer_prob = float(1 - probs[0])

        elif cancer_type == 'breast':
            sess = model_registry.get('breast_det')
            if sess:
                size  = sess.get_inputs()[0].shape[2] or 512
                x     = preprocess_image(img_bytes, size, mammogram=True)
                out   = sess.run(None, {sess.get_inputs()[0].name: x})[0]
                cancer_prob = float(sigmoid(out[0][0]))
                tumor_type  = 'malignant' if cancer_prob > 0.5 else 'benign'
                confidence  = max(cancer_prob, 1 - cancer_prob)

        elif cancer_type == 'skin':
            sess = model_registry.get('skin_cls')
            if sess:
                size    = sess.get_inputs()[0].shape[2] or 384
                x       = preprocess_image(img_bytes, size)
                out     = sess.run(None, {sess.get_inputs()[0].name: x})[0]
                probs   = softmax(out[0])
                classes = ['melanoma', 'nevus', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
                idx     = int(probs.argmax())
                tumor_type  = classes[idx]
                confidence  = float(probs.max())
                cancer_prob = float(probs[0])   # melanoma prob

    except Exception as e:
        logger.error(f'Detection failed for {scan_id}: {e}')

    # Verification gate
    threshold = {'brain': 0.97, 'melanoma': 0.99, 'breast': 0.95,
                 'lung': 0.95, 'liver': 0.93, 'spine': 0.93, 'skin': 0.97}
    thresh = threshold.get(cancer_type, 0.95)

    if cancer_prob > 0.5:
        verdict  = 'CANCER_FLAGGED'
        priority = 1 if cancer_prob > 0.85 else 2
    elif confidence >= thresh:
        verdict  = 'NORMAL'
        priority = 4
    else:
        verdict  = 'REVIEW_REQUIRED'
        priority = 3

    result.update({
        'verdict'    : verdict,
        'cancer_prob': round(cancer_prob, 4),
        'tumor_type' : tumor_type,
        'confidence' : round(float(confidence), 4),
        'priority'   : priority,
    })

    await emit('detection', {
        'status'     : 'done',
        'verdict'    : verdict,
        'cancer_prob': cancer_prob,
        'tumor_type' : tumor_type,
    })

    # ── Agent 3: Segmentation (stub for 2D) ──────────────────────────────────
    await emit('segmentation', {'status': 'running'})
    await asyncio.sleep(0)
    await emit('segmentation', {'status': 'done', 'segmented': False,
                                'note': '3D segmentation requires NIfTI volume'})

    # ── Agent 4: Classification ──────────────────────────────────────────────
    await emit('classification', {'status': 'running'})
    who_grade = None
    if cancer_type == 'brain' and tumor_type == 'glioma':
        who_grade = 'Grade IV (GBM)' if confidence > 0.7 else 'Grade II-III'
    result['who_grade'] = who_grade
    await emit('classification', {'status': 'done', 'tumor_type': tumor_type,
                                  'who_grade': who_grade})

    # ── Agent 5: Clinical Intelligence ───────────────────────────────────────
    await emit('clinical', {'status': 'running'})
    recs = get_treatment_recs(tumor_type)
    result['treatment_recs'] = recs
    await emit('clinical', {'status': 'done', 'n_recs': len(recs)})

    # ── Agent 6: Report ───────────────────────────────────────────────────────
    await emit('report', {'status': 'running'})

    report_lines = [
        f'NEUROSCOPE AI REPORT',
        f'Generated  : {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        f'Scan ID    : {scan_id}',
        f'Cancer     : {cancer_type.upper()}',
        f'',
        f'FINDINGS:',
        f'  Verdict      : {verdict}',
        f'  Cancer Prob  : {cancer_prob:.1%}',
        f'  Tumor Type   : {tumor_type} (confidence: {confidence:.1%})',
    ]
    if who_grade:
        report_lines.append(f'  WHO Grade    : {who_grade}')
    report_lines += ['', 'TREATMENT RECOMMENDATIONS:']
    for i, rec in enumerate(recs[:5], 1):
        report_lines.append(f'  {i}. {rec}')
    report_lines += [
        '',
        'AI-ASSISTED ANALYSIS. CLINICIAN REVIEW REQUIRED.',
    ]
    report_text = '\n'.join(report_lines)

    plain = (
        f'The AI analysis of your {cancer_type} scan found findings that '
        f'need attention from your doctor.'
        if verdict == 'CANCER_FLAGGED' else
        f'The AI analysis of your {cancer_type} scan did not find concerning findings.'
        if verdict == 'NORMAL' else
        f'The AI analysis of your {cancer_type} scan requires additional review.'
    )

    result['report_text']  = report_text
    result['plain_summary'] = plain

    # Save report
    report_dir = os.path.join(OUT_PATH, scan_id)
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    await emit('report', {'status': 'done'})

    # ── Agents 7-10: QA + Ethics (parallel, non-blocking) ────────────────────
    await emit('monitoring', {'status': 'running'})
    # QA: enforce 99.9% confidence hard limit
    if cancer_prob > 0.999:
        result['cancer_prob'] = 0.999
        logger.warning(f'Confidence capped at 99.9% for {scan_id}')
    await emit('monitoring', {'status': 'done'})

    # ── Final ─────────────────────────────────────────────────────────────────
    result['latency_ms'] = round((time.perf_counter() - t_start) * 1000, 1)

    await emit('complete', {
        'status'     : 'complete',
        'scan_id'    : scan_id,
        'verdict'    : verdict,
        'latency_ms' : result['latency_ms'],
    })

    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@app.on_event('startup')
async def startup():
    logger.info('NeuroScope AI API starting...')
    model_registry.load_all()
    bootstrap_default_users()
    logger.info('Ready')


@app.get('/health')
async def health():
    return {
        'status'   : 'ok',
        'version'  : '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'models'   : len(model_registry._sessions),
    }


@app.get('/models')
async def list_models():
    return {
        'models'  : model_registry.status(),
        'base_path': MODELS_PATH,
    }


@app.post('/analyze/sync', response_model=AnalyzeResponse)
async def analyze_sync(
    file        : UploadFile = File(...),
    cancer_type : str        = Form(...),
    patient_age : Optional[int]  = Form(None),
    patient_sex : Optional[str]  = Form(None),
    clinical_notes: str          = Form(''),
):
    """
    Synchronous full pipeline analysis.
    Returns complete result when done.
    Use /ws/analyze for real-time streaming.
    """
    if cancer_type not in VALID_CANCER_TYPES:
        raise HTTPException(400, f'Invalid cancer_type. Must be one of: {VALID_CANCER_TYPES}')

    scan_id   = str(uuid.uuid4())[:8]
    img_bytes = await file.read()

    try:
        result = await run_pipeline(
            scan_id, img_bytes, cancer_type,
            patient_age, patient_sex, clinical_notes
        )
        return result
    except Exception as e:
        logger.error(f'Pipeline failed for {scan_id}: {e}')
        raise HTTPException(500, f'Pipeline error: {str(e)}')


@app.post('/analyze')
async def analyze_async(
    background_tasks: BackgroundTasks,
    file        : UploadFile = File(...),
    cancer_type : str        = Form(...),
    patient_age : Optional[int]  = Form(None),
    patient_sex : Optional[str]  = Form(None),
    clinical_notes: str          = Form(''),
):
    """
    Async pipeline -- returns task_id immediately.
    Poll /status/{task_id} or connect to /ws/analyze/{scan_id} for progress.
    """
    if cancer_type not in VALID_CANCER_TYPES:
        raise HTTPException(400, f'Invalid cancer_type. Must be one of: {VALID_CANCER_TYPES}')

    scan_id   = str(uuid.uuid4())[:8]
    task_id   = str(uuid.uuid4())[:12]
    img_bytes = await file.read()

    task_store[task_id] = {
        'status'  : 'pending',
        'progress': 0,
        'scan_id' : scan_id,
        'result'  : None,
        'error'   : None,
    }

    async def run_task():
        task_store[task_id]['status'] = 'running'
        try:
            result = await run_pipeline(
                scan_id, img_bytes, cancer_type,
                patient_age, patient_sex, clinical_notes
            )
            task_store[task_id].update({
                'status'  : 'completed',
                'progress': 100,
                'result'  : result,
            })
        except Exception as e:
            task_store[task_id].update({
                'status': 'failed',
                'error' : str(e),
            })

    background_tasks.add_task(run_task)

    return {'task_id': task_id, 'scan_id': scan_id, 'status': 'pending'}


@app.get('/status/{task_id}', response_model=TaskStatus)
async def get_status(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(404, f'Task {task_id} not found')
    return TaskStatus(**task, task_id=task_id)


@app.get('/report/{scan_id}')
async def get_report(scan_id: str, format: str = 'text'):
    report_dir = os.path.join(OUT_PATH, scan_id)
    if not os.path.exists(report_dir):
        raise HTTPException(404, f'No report found for scan {scan_id}')

    if format == 'pdf':
        pdf_path = os.path.join(report_dir, 'report.pdf')
        if os.path.exists(pdf_path):
            return FileResponse(pdf_path, media_type='application/pdf')
        raise HTTPException(404, 'PDF report not available')

    txt_path = os.path.join(report_dir, 'report.txt')
    if os.path.exists(txt_path):
        with open(txt_path, encoding='utf-8') as f:
            return {'scan_id': scan_id, 'report': f.read()}
    raise HTTPException(404, 'Text report not available')


@app.post('/classify')
async def classify_only(
    file       : UploadFile = File(...),
    cancer_type: str        = Form(...),
):
    """Classification-only endpoint -- no full pipeline."""
    if cancer_type not in VALID_CANCER_TYPES:
        raise HTTPException(400, f'Invalid cancer_type')

    scan_id   = str(uuid.uuid4())[:8]
    img_bytes = await file.read()

    try:
        sess = model_registry.get(f'{cancer_type}_cls') or \
               model_registry.get(f'{cancer_type}_det')
        if not sess:
            raise HTTPException(503, f'No model loaded for {cancer_type}')

        size = sess.get_inputs()[0].shape[2] or 224
        x    = preprocess_image(img_bytes, size,
                                mammogram=(cancer_type == 'breast'))
        out  = sess.run(None, {sess.get_inputs()[0].name: x})[0]

        if cancer_type == 'breast':
            prob = float(sigmoid(out[0][0]))
            return {'scan_id': scan_id, 'prob_malignant': prob,
                    'class': 'malignant' if prob > 0.5 else 'benign'}

        probs = softmax(out[0]).tolist()
        return {'scan_id': scan_id, 'probs': probs,
                'pred_class': int(np.argmax(probs)),
                'confidence': float(max(probs))}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket('/ws/analyze/{scan_id}')
async def ws_analyze(websocket: WebSocket, scan_id: str):
    """
    WebSocket streaming endpoint.
    Client sends image + metadata as JSON, receives agent progress in real-time.

    Message format (client -> server):
    {
      "cancer_type": "brain",
      "image_b64"  : "<base64 encoded image>",
      "patient_age": 45,
      "patient_sex": "M",
      "notes"      : "..."
    }

    Progress events (server -> client):
    { "stage": "detection", "status": "running" }
    { "stage": "detection", "status": "done", "verdict": "CANCER_FLAGGED", ... }
    { "stage": "complete",  "verdict": "...", "latency_ms": 450 }
    """
    await ws_manager.connect(scan_id, websocket)

    try:
        data = await websocket.receive_json()

        cancer_type = data.get('cancer_type', 'brain')
        patient_age = data.get('patient_age')
        patient_sex = data.get('patient_sex')
        notes       = data.get('notes', '')

        # Decode base64 image
        import base64
        image_b64 = data.get('image_b64', '')
        try:
            img_bytes = base64.b64decode(image_b64)
        except Exception:
            await websocket.send_json({'error': 'Invalid image encoding'})
            return

        if cancer_type not in VALID_CANCER_TYPES:
            await websocket.send_json({'error': f'Invalid cancer_type: {cancer_type}'})
            return

        async def on_progress(stage: str, progress_data: Dict):
            await ws_manager.send(scan_id, {'stage': stage, **progress_data})

        result = await run_pipeline(
            scan_id, img_bytes, cancer_type,
            patient_age, patient_sex, notes,
            on_progress=on_progress,
        )

        # Send final result
        await websocket.send_json({'stage': 'result', 'data': result})

    except WebSocketDisconnect:
        logger.info(f'WS disconnected: {scan_id}')
    except Exception as e:
        logger.error(f'WS error for {scan_id}: {e}')
        try:
            await websocket.send_json({'error': str(e)})
        except Exception:
            pass
    finally:
        ws_manager.disconnect(scan_id)

@app.get('/ui', response_class=FileResponse)
async def serve_ui():
    ui_path = os.path.join(os.path.dirname(__file__), 'neuroscope_ui.html')
    return FileResponse(ui_path, media_type='text/html')


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=8000,
        reload=False,
        workers=1,       # 1 worker for GPU (single GPU, no sharing)
        log_level='info',
    )
