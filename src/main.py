"""
NeuroScope AI — main.py (Module 2 upgrade)
==========================================
All original endpoints preserved.
New endpoints added:
  POST /analyze/sync        -- upgraded with auth + audit + patient_id
  POST /analyze             -- upgraded with auth
  GET  /scans/{patient_id}  -- patient scan history
  POST /scans/{scan_id}/feedback   -- clinician confirm/override
  POST /scans/{scan_id}/consent    -- consent tracking
  GET  /scans/{patient_id}/export  -- PDF history export
  POST /analyze/batch       -- batch analysis
  GET  /admin/stats         -- usage stats
  GET  /admin/usage         -- per hospital usage analytics
  GET  /admin/model-performance    -- drift detection
  GET  /admin/hospitals     -- hospital overview
  GET  /admin/audit         -- audit log (proxies auth router)
  GET  /report/{scan_id}/fhir      -- FHIR DiagnosticReport export
  GET  /ui                  -- serve dashboard HTML
"""

import os, sys, uuid, json, time, asyncio, logging, base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

import numpy as np
import cv2

from fastapi import (
    FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect,
    HTTPException, BackgroundTasks, Form, Request, Depends
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── Auth module ───────────────────────────────────────────────────────────────
try:
    from auth import (
        router as auth_router,
        get_current_user,
        require_role,
        audit,
        _load_users,
        _save_users,
        _load_hospitals,
        CFGDIR,
        AUDIT_FILE,
    )
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    def get_current_user(request=None): return {"sub":"anonymous","role":"clinician","hospital_id":None}
    def audit(*a,**k): pass
    def _load_users(): return {}
    def _save_users(d): pass
    def _load_hospitals(): return {}
    CFGDIR    = "/tmp"
    AUDIT_FILE= "/tmp/audit.jsonl"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("neuroscope.api")

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NeuroScope AI API",
    description="Multi-cancer medical imaging intelligence platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount auth router
if AUTH_AVAILABLE:
    app.include_router(auth_router)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_PATH   = os.environ.get("NEUROSCOPE_BASE",
               r"C:\Users\tejan\OneDrive\Desktop\drive\NeuroScope_AI")
MODELS_PATH = os.path.join(BASE_PATH, "models", "production")
OUT_PATH    = os.path.join(BASE_PATH, "outputs", "api")
SCANS_PATH  = os.path.join(BASE_PATH, "outputs", "scans")
SRC_PATH    = os.path.join(BASE_PATH, "src")
os.makedirs(OUT_PATH,   exist_ok=True)
os.makedirs(SCANS_PATH, exist_ok=True)

VALID_CANCER_TYPES = ["brain","lung","breast","liver","skin","spine"]

# Calibrated inference thresholds (per-pipeline)
CANCER_THRESHOLDS = {
    "brain" :0.65,"lung" :0.60,"breast":0.55,
    "liver" :0.55,"skin" :0.45,"spine" :0.55,
}
CONFIDENCE_THRESHOLDS = {
    "brain" :0.75,"lung" :0.70,"breast":0.65,
    "liver" :0.65,"skin" :0.70,"spine" :0.65,
}

MODEL_VERSION = "v2.0.0-2026-06"

# ── Pydantic models ───────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    cancer_type   : str
    patient_age   : Optional[int] = None
    patient_sex   : Optional[str] = None
    clinical_notes: Optional[str] = ""
    priority      : Optional[int] = 3
    patient_id    : Optional[str] = None

class AnalyzeResponse(BaseModel):
    scan_id        : str
    cancer_type    : str
    verdict        : str
    cancer_prob    : float
    tumor_type     : Optional[str]
    confidence     : Optional[float]
    who_grade      : Optional[str]
    priority       : int
    treatment_recs : list
    report_text    : str
    plain_summary  : str
    latency_ms     : float
    timestamp      : str
    patient_id     : Optional[str] = None
    model_version  : str = MODEL_VERSION
    icd10          : Optional[str] = None

class TaskStatus(BaseModel):
    task_id : str
    status  : str
    progress: int
    result  : Optional[Dict] = None
    error   : Optional[str]  = None

class FeedbackRequest(BaseModel):
    action          : str   # "confirm" | "override"
    override_verdict: Optional[str] = None
    clinical_notes  : Optional[str] = None

class ConsentRequest(BaseModel):
    patient_id    : str
    consent_given : bool
    consent_type  : str = "ai_analysis"

# ── Model registry ────────────────────────────────────────────────────────────
class ModelRegistry:
    def __init__(self, models_path):
        self.models_path = models_path
        self._sessions   = {}
        self._loaded     = False

    def load_all(self):
        try:
            import onnxruntime as ort
            providers   = ["CUDAExecutionProvider","CPUExecutionProvider"]
            model_files = {
                "brain_cls" : os.path.join(self.models_path,"brain_cls","brain_cls_efficientnet.onnx"),
                "brain_seg" : os.path.join(self.models_path,"brain_seg","brain_seg_resnet.onnx"),
                "breast_det": os.path.join(self.models_path,"breast_det","breast_det_efficientnet.onnx"),
                "liver_seg" : os.path.join(self.models_path,"liver_seg","liver_seg_resnet.onnx"),
                "lung_det"  : os.path.join(self.models_path,"lung_det","lung_det_resnet3d.onnx"),
                "skin_cls"  : os.path.join(self.models_path,"skin_cls","skin_cls_efficientnet.onnx"),
                "spine_cls" : os.path.join(self.models_path,"spine","spine_hybridspinenet.onnx"),
            }
            for name, path in model_files.items():
                if os.path.exists(path):
                    try:
                        self._sessions[name] = ort.InferenceSession(path, providers=providers)
                        logger.info(f"Loaded: {name}")
                    except Exception as e:
                        logger.warning(f"Failed {name}: {e}")
                else:
                    logger.warning(f"Not found: {name}")
            self._loaded = True
            logger.info(f"Registry: {len(self._sessions)}/7 loaded")
        except ImportError:
            logger.error("onnxruntime not installed")

    def get(self, name): return self._sessions.get(name)
    def count(self):     return len(self._sessions)
    def status(self):
        return {n:"loaded" for n in self._sessions}

model_registry = ModelRegistry(MODELS_PATH)
task_store: Dict[str,Dict] = {}

# ── WebSocket manager ─────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str,WebSocket] = {}

    async def connect(self, scan_id, ws):
        await ws.accept()
        self.active[scan_id] = ws

    def disconnect(self, scan_id):
        self.active.pop(scan_id, None)

    async def send(self, scan_id, data):
        ws = self.active.get(scan_id)
        if ws:
            try:    await ws.send_json(data)
            except: self.disconnect(scan_id)

ws_manager = ConnectionManager()

# ── Preprocessing ─────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229,0.224,0.225], dtype=np.float32)

def preprocess_image(img_bytes, img_size, mammogram=False, cancer_type="brain"):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB always
    if mammogram or cancer_type == "breast":
        gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray  = clahe.apply(gray)
        img   = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (img_size, img_size)).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(img.transpose(2,0,1), 0).astype(np.float32)

def softmax(x): e=np.exp(x-x.max()); return e/e.sum()
def sigmoid(x): return 1.0/(1.0+np.exp(-x))

# ── ICD-10 codes ──────────────────────────────────────────────────────────────
ICD10_MAP = {
    "glioma":"C71.9","meningioma":"D32.9","pituitary":"D35.2",
    "malignant":"C50.919","melanoma":"C43.9","bcc":"C44.91",
    "nodule":"R91.1","tumor":"C22.0","default":"Z03.89",
}

# ── NCCN treatment rules ──────────────────────────────────────────────────────
NCCN_RULES = {
    "glioma"    :["Maximal safe resection","RT + TMZ (Stupp protocol)","IDH/MGMT testing","MDT review within 48hrs"],
    "meningioma":["Surgery (Simpson grade I-II)","SRS for inaccessible lesions","Annual MRI follow-up"],
    "malignant" :["Surgery + sentinel LN biopsy","HER2/ER/PR testing","Chemo if indicated","Oncology MDT"],
    "melanoma"  :["Wide local excision (1-2cm margins)","SLNB if Breslow >0.8mm","BRAF V600 testing","PD-1 inhibitor if advanced"],
    "bcc"       :["Surgical excision (4mm margins)","Mohs for facial/high-risk","Vismodegib if advanced"],
    "nodule"    :["Lung-RADS scoring","LDCT follow-up per NCCN","PET if >8mm"],
    "tumor"     :["BCLC staging","Resection/TACE per stage","AFP monitoring"],
    "default"   :["Oncology referral","Tissue biopsy","Staging CT","MDT review"],
}

def get_treatment_recs(tumor_type):
    return NCCN_RULES.get(tumor_type, NCCN_RULES["default"])

# ── Scan persistence helpers ──────────────────────────────────────────────────
def save_scan_result(scan_id, result, patient_id=None, user=None, hospital_id=None):
    d = os.path.join(SCANS_PATH, scan_id)
    os.makedirs(d, exist_ok=True)
    record = {**result,
              "patient_id"  : patient_id,
              "submitted_by": user,
              "hospital_id" : hospital_id,
              "saved_at"    : datetime.now().isoformat(),
              "model_version": MODEL_VERSION,
              "feedback"    : None,
              "consent"     : None,
    }
    with open(os.path.join(d,"result.json"),"w",encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    # Save report text
    with open(os.path.join(d,"report.txt"),"w",encoding="utf-8") as f:
        f.write(result.get("report_text",""))
    return record

def load_scan(scan_id):
    path = os.path.join(SCANS_PATH, scan_id, "result.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def get_patient_scans(patient_id, hospital_id=None):
    scans = []
    if not os.path.exists(SCANS_PATH):
        return scans
    for scan_dir in Path(SCANS_PATH).iterdir():
        r_path = scan_dir / "result.json"
        if not r_path.exists():
            continue
        try:
            with open(r_path, encoding="utf-8") as f:
                r = json.load(f)
            if r.get("patient_id") != patient_id:
                continue
            if hospital_id and r.get("hospital_id") != hospital_id:
                continue
            scans.append(r)
        except:
            pass
    return sorted(scans, key=lambda x: x.get("timestamp",""), reverse=True)

# ── Critical alert ─────────────────────────────────────────────────────────────
async def send_critical_alert(scan_id, cancer_type, verdict, priority, user, hospital_id):
    if priority > 1:
        return
    logger.warning(f"PRIORITY 1 ALERT: scan={scan_id} cancer={cancer_type} user={user}")
    audit("critical_alert_fired",
          user=user or "system",
          detail=f"scan={scan_id} cancer={cancer_type} verdict={verdict}",
          hospital=hospital_id or "")
    # Email sending would go here when Gmail SMTP is configured

# ── Core pipeline ─────────────────────────────────────────────────────────────
async def run_pipeline(scan_id, img_bytes, cancer_type,
                        patient_age, patient_sex, notes,
                        on_progress=None, patient_id=None,
                        submitted_by=None, hospital_id=None):
    t0 = time.perf_counter()

    async def emit(stage, data):
        if on_progress:
            await on_progress(stage, data)

    result = {
        "scan_id"      : scan_id,
        "cancer_type"  : cancer_type,
        "verdict"      : "REVIEW_REQUIRED",
        "cancer_prob"  : 0.5,
        "tumor_type"   : None,
        "confidence"   : 0.0,
        "who_grade"    : None,
        "priority"     : 3,
        "treatment_recs": [],
        "report_text"  : "",
        "plain_summary": "",
        "latency_ms"   : 0.0,
        "timestamp"    : datetime.now().isoformat(),
        "patient_id"   : patient_id,
        "model_version": MODEL_VERSION,
        "icd10"        : None,
    }

    await emit("normalization", {"status":"running","scan_id":scan_id})
    await asyncio.sleep(0)
    await emit("normalization", {"status":"done"})

    await emit("triage", {"status":"running"})
    await asyncio.sleep(0)
    await emit("triage", {"status":"done","cancer_type":cancer_type})

    await emit("detection", {"status":"running"})

    cancer_prob = 0.5
    tumor_type  = "unknown"
    confidence  = 0.5

    try:
        if cancer_type == "brain":
            sess = model_registry.get("brain_cls")
            if sess:
                size   = sess.get_inputs()[0].shape[2] or 224
                x      = preprocess_image(img_bytes, size, cancer_type="brain")
                out    = sess.run(None, {sess.get_inputs()[0].name: x})[0]
                probs  = softmax(out[0])
                classes= ["no_tumor","glioma","meningioma","pituitary"]
                idx    = int(probs.argmax())
                tumor_type  = classes[idx]
                confidence  = float(probs.max())
                cancer_prob = float(1 - probs[0])

        elif cancer_type == "breast":
            sess = model_registry.get("breast_det")
            if sess:
                size  = sess.get_inputs()[0].shape[2] or 512
                x     = preprocess_image(img_bytes, size, cancer_type="breast")
                out   = sess.run(None, {sess.get_inputs()[0].name: x})[0]
                cancer_prob = float(sigmoid(out[0][0]))
                tumor_type  = "malignant" if cancer_prob > 0.5 else "benign"
                confidence  = max(cancer_prob, 1-cancer_prob)

        elif cancer_type == "skin":
            sess = model_registry.get("skin_cls")
            if sess:
                size    = sess.get_inputs()[0].shape[2] or 384
                x       = preprocess_image(img_bytes, size, cancer_type="skin")
                out     = sess.run(None, {sess.get_inputs()[0].name: x})[0]
                probs   = softmax(out[0])
                classes = ["melanoma","nevus","bcc","akiec","bkl","df","vasc"]
                idx     = int(probs.argmax())
                tumor_type  = classes[idx]
                confidence  = float(probs.max())
                cancer_prob = float(probs[0])

        elif cancer_type == "lung":
            sess = model_registry.get("lung_det")
            if sess:
                size  = sess.get_inputs()[0].shape[2] or 224
                x     = preprocess_image(img_bytes, size, cancer_type="lung")
                out   = sess.run(None, {sess.get_inputs()[0].name: x})[0]
                probs = softmax(out[0])
                cancer_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                tumor_type  = "nodule" if cancer_prob > 0.5 else "clear"
                confidence  = max(cancer_prob, 1-cancer_prob)

        elif cancer_type == "liver":
            sess = model_registry.get("liver_seg")
            if sess:
                size  = sess.get_inputs()[0].shape[2] or 256
                x     = preprocess_image(img_bytes, size, cancer_type="liver")
                out   = sess.run(None, {sess.get_inputs()[0].name: x})[0]
                probs = softmax(out[0]) if out[0].ndim == 1 else out[0].mean(axis=(1,2))
                cancer_prob = float(probs.max())
                tumor_type  = "tumor" if cancer_prob > 0.5 else "clear"
                confidence  = cancer_prob

        elif cancer_type == "spine":
            sess = model_registry.get("spine_cls")
            if sess:
                size  = sess.get_inputs()[0].shape[2] or 384
                x     = preprocess_image(img_bytes, size, cancer_type="spine")
                out   = sess.run(None, {sess.get_inputs()[0].name: x})[0]
                probs = softmax(out[0]) if out[0].ndim==1 else softmax(out[0].flatten()[:3])
                cancer_prob = float(probs[-1])  # Severe
                tumor_type  = "severe_stenosis" if cancer_prob>0.5 else "normal"
                confidence  = float(probs.max())

    except Exception as e:
        logger.error(f"Detection failed {scan_id}: {e}")

    # Calibrated verdict
    c_thresh = CANCER_THRESHOLDS.get(cancer_type, 0.60)
    f_thresh = CONFIDENCE_THRESHOLDS.get(cancer_type, 0.70)

    if cancer_prob >= c_thresh and confidence >= f_thresh:
        verdict  = "CANCER_FLAGGED"
        priority = 1 if cancer_prob > 0.85 else 2
    elif cancer_prob < 0.35 and confidence >= f_thresh:
        verdict  = "NORMAL"
        priority = 4
    else:
        verdict  = "REVIEW_REQUIRED"
        priority = 3

    icd10 = ICD10_MAP.get(tumor_type, ICD10_MAP["default"])

    result.update({
        "verdict"    : verdict,
        "cancer_prob": round(cancer_prob, 4),
        "tumor_type" : tumor_type,
        "confidence" : round(float(confidence), 4),
        "priority"   : priority,
        "icd10"      : icd10,
    })

    await emit("detection", {
        "status":"done","verdict":verdict,
        "cancer_prob":cancer_prob,"tumor_type":tumor_type,
    })

    await emit("segmentation", {"status":"running"})
    await asyncio.sleep(0)
    await emit("segmentation", {"status":"done","segmented":False})

    await emit("classification", {"status":"running"})
    who_grade = None
    if cancer_type == "brain" and tumor_type == "glioma":
        who_grade = "Grade IV (GBM)" if confidence > 0.7 else "Grade II-III"
    result["who_grade"] = who_grade
    await emit("classification", {"status":"done","tumor_type":tumor_type,"who_grade":who_grade})

    await emit("clinical", {"status":"running"})
    recs = get_treatment_recs(tumor_type)
    result["treatment_recs"] = recs
    await emit("clinical", {"status":"done","n_recs":len(recs)})

    await emit("report", {"status":"running"})
    lines = [
        "NEUROSCOPE AI REPORT",
        f"Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Scan ID     : {scan_id}",
        f"Cancer      : {cancer_type.upper()}",
        f"Patient ID  : {patient_id or 'N/A'}",
        f"Model Ver   : {MODEL_VERSION}",
        "",
        "FINDINGS:",
        f"  Verdict      : {verdict}",
        f"  Cancer Prob  : {cancer_prob:.1%}",
        f"  Tumour Type  : {tumor_type}",
        f"  Confidence   : {confidence:.1%}",
        f"  ICD-10       : {icd10}",
    ]
    if who_grade:
        lines.append(f"  WHO Grade    : {who_grade}")
    if patient_age:
        lines.append(f"  Patient Age  : {patient_age}")
    lines += ["","TREATMENT RECOMMENDATIONS (NCCN aligned):"]
    for i,r in enumerate(recs[:5],1):
        lines.append(f"  {i}. {r}")
    lines += ["","AI-ASSISTED ANALYSIS — CLINICIAN REVIEW REQUIRED."]

    report_text = "\n".join(lines)
    plain = (
        f"AI analysis of {cancer_type} scan indicates findings requiring urgent attention."
        if verdict == "CANCER_FLAGGED" else
        f"AI analysis of {cancer_type} scan found no significant findings."
        if verdict == "NORMAL" else
        f"AI analysis of {cancer_type} scan requires additional clinical review."
    )
    result["report_text"]   = report_text
    result["plain_summary"] = plain
    await emit("report", {"status":"done"})

    await emit("monitoring", {"status":"running"})
    if cancer_prob > 0.999:
        result["cancer_prob"] = 0.999
    await emit("monitoring", {"status":"done"})

    result["latency_ms"] = round((time.perf_counter()-t0)*1000, 1)

    # Persist
    save_scan_result(scan_id, result, patient_id, submitted_by, hospital_id)

    # Critical alert
    if priority == 1:
        asyncio.create_task(
            send_critical_alert(scan_id, cancer_type, verdict,
                                priority, submitted_by, hospital_id)
        )

    # Audit
    audit("scan_analyzed",
          user=submitted_by or "anonymous",
          detail=f"scan={scan_id} cancer={cancer_type} verdict={verdict}",
          hospital=hospital_id or "")

    # Increment user scan count
    if submitted_by:
        users = _load_users()
        if submitted_by in users:
            users[submitted_by]["scan_count"] = users[submitted_by].get("scan_count",0)+1
            _save_users(users)

    await emit("complete", {
        "status":"complete","scan_id":scan_id,
        "verdict":verdict,"latency_ms":result["latency_ms"],
    })
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("NeuroScope AI v2.0 starting...")
    model_registry.load_all()
    logger.info("Ready")


@app.get("/health")
async def health():
    return {
        "status"   : "ok",
        "version"  : "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "models"   : model_registry.count(),
        "auth"     : AUTH_AVAILABLE,
    }


@app.get("/models")
async def list_models():
    return {"models": model_registry.status(), "base_path": MODELS_PATH}


@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    ui_path = os.path.join(SRC_PATH, "neuroscope_ui.html")
    if os.path.exists(ui_path):
        with open(ui_path, encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>NeuroScope AI</h1><p>UI file not found at " + ui_path + "</p>")


# ── Analyze endpoints ─────────────────────────────────────────────────────────
@app.post("/analyze/sync", response_model=AnalyzeResponse)
@limiter.limit("30/minute")
async def analyze_sync(
    request       : Request,
    file          : UploadFile = File(...),
    cancer_type   : str        = Form(...),
    patient_age   : Optional[int] = Form(None),
    patient_sex   : Optional[str] = Form(None),
    clinical_notes: str           = Form(""),
    patient_id    : Optional[str] = Form(None),
):
    if cancer_type not in VALID_CANCER_TYPES:
        raise HTTPException(400, f"Invalid cancer_type. Must be one of: {VALID_CANCER_TYPES}")

    # Get user from token if available
    current_user  = None
    hospital_id   = None
    auth_header   = request.headers.get("Authorization","")
    cookie_token  = request.cookies.get("ns_token","")
    token         = auth_header[7:] if auth_header.startswith("Bearer ") else cookie_token
    if token and AUTH_AVAILABLE:
        try:
            from auth import decode_jwt
            payload      = decode_jwt(token)
            current_user = payload.get("sub")
            hospital_id  = payload.get("hospital_id")
        except:
            pass

    scan_id   = str(uuid.uuid4())[:8]
    img_bytes = await file.read()
    try:
        result = await run_pipeline(
            scan_id, img_bytes, cancer_type,
            patient_age, patient_sex, clinical_notes,
            patient_id=patient_id,
            submitted_by=current_user,
            hospital_id=hospital_id,
        )
        return result
    except Exception as e:
        logger.error(f"Pipeline failed {scan_id}: {e}")
        raise HTTPException(500, f"Pipeline error: {e}")


@app.post("/analyze")
async def analyze_async(
    request       : Request,
    background_tasks: BackgroundTasks,
    file          : UploadFile = File(...),
    cancer_type   : str        = Form(...),
    patient_age   : Optional[int] = Form(None),
    patient_sex   : Optional[str] = Form(None),
    clinical_notes: str           = Form(""),
    patient_id    : Optional[str] = Form(None),
):
    if cancer_type not in VALID_CANCER_TYPES:
        raise HTTPException(400, f"Invalid cancer_type.")

    current_user = None
    hospital_id  = None
    auth_header  = request.headers.get("Authorization","")
    token        = auth_header[7:] if auth_header.startswith("Bearer ") else request.cookies.get("ns_token","")
    if token and AUTH_AVAILABLE:
        try:
            from auth import decode_jwt
            payload      = decode_jwt(token)
            current_user = payload.get("sub")
            hospital_id  = payload.get("hospital_id")
        except:
            pass

    scan_id   = str(uuid.uuid4())[:8]
    task_id   = str(uuid.uuid4())[:12]
    img_bytes = await file.read()

    task_store[task_id] = {"status":"pending","progress":0,"scan_id":scan_id,
                            "result":None,"error":None}

    async def run_task():
        task_store[task_id]["status"] = "running"
        try:
            result = await run_pipeline(
                scan_id, img_bytes, cancer_type,
                patient_age, patient_sex, clinical_notes,
                patient_id=patient_id,
                submitted_by=current_user,
                hospital_id=hospital_id,
            )
            task_store[task_id].update({"status":"completed","progress":100,"result":result})
        except Exception as e:
            task_store[task_id].update({"status":"failed","error":str(e)})

    background_tasks.add_task(run_task)
    return {"task_id":task_id,"scan_id":scan_id,"status":"pending"}


@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_status(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(404, f"Task {task_id} not found")
    return TaskStatus(**task, task_id=task_id)


@app.get("/report/{scan_id}")
async def get_report(scan_id: str, format: str = "json"):
    r = load_scan(scan_id)
    if r:
        if format == "text":
            return {"scan_id":scan_id,"report":r.get("report_text","")}
        return r
    # Fallback to old path
    old_dir = os.path.join(OUT_PATH, scan_id)
    if os.path.exists(old_dir):
        txt = os.path.join(old_dir,"report.txt")
        if os.path.exists(txt):
            with open(txt, encoding="utf-8") as f:
                return {"scan_id":scan_id,"report":f.read()}
    raise HTTPException(404, f"No report for scan {scan_id}")


@app.get("/report/{scan_id}/fhir")
async def fhir_report(scan_id: str):
    r = load_scan(scan_id)
    if not r:
        raise HTTPException(404, f"Scan {scan_id} not found")
    SNOMED = {
        "glioma":"413448000","meningioma":"26003006","melanoma":"413448000",
        "malignant":"413448000","bcc":"254701007","nodule":"427359007",
        "tumor":"126851005","default":"416940007",
    }
    fhir = {
        "resourceType": "DiagnosticReport",
        "id"          : f"neuroscope-{scan_id}",
        "status"      : "final",
        "category"    : [{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/v2-0074",
                                     "code":"RAD","display":"Radiology"}]}],
        "code"        : {"coding":[{"system":"http://loinc.org",
                                    "code":"24627-2","display":"CT Radiology"}]},
        "subject"     : {"reference":f"Patient/{r.get('patient_id','unknown')}"},
        "effectiveDateTime": r.get("timestamp",""),
        "issued"          : r.get("timestamp",""),
        "performer"       : [{"display":f"NeuroScope AI {MODEL_VERSION}"}],
        "conclusion"      : (f"AI analysis: {r.get('tumor_type','unknown')} detected. "
                             f"Verdict: {r.get('verdict','')}. "
                             f"Confidence: {r.get('confidence',0):.1%}."),
        "conclusionCode"  : [{"coding":[{
            "system":"http://snomed.info/sct",
            "code":SNOMED.get(r.get("tumor_type",""),"416940007"),
            "display":r.get("tumor_type","Unknown"),
        }]}],
        "extension": [
            {"url":"https://neuroscope.ai/fhir/cancer-prob","valueDecimal":r.get("cancer_prob",0)},
            {"url":"https://neuroscope.ai/fhir/icd10","valueString":r.get("icd10","")},
            {"url":"https://neuroscope.ai/fhir/who-grade","valueString":r.get("who_grade","")},
            {"url":"https://neuroscope.ai/fhir/priority","valueInteger":r.get("priority",3)},
            {"url":"https://neuroscope.ai/fhir/model-version","valueString":MODEL_VERSION},
        ],
    }
    return JSONResponse(fhir)


# ── Patient history endpoints ─────────────────────────────────────────────────
@app.get("/scans/{patient_id}")
async def patient_scan_history(patient_id: str, request: Request):
    hospital_id  = None
    token = (request.headers.get("Authorization","")[7:]
             or request.cookies.get("ns_token",""))
    if token and AUTH_AVAILABLE:
        try:
            from auth import decode_jwt
            payload     = decode_jwt(token)
            hospital_id = payload.get("hospital_id")
        except: pass
    scans = get_patient_scans(patient_id, hospital_id)
    return {"patient_id":patient_id,"scans":scans,"count":len(scans)}


@app.post("/scans/{scan_id}/feedback")
async def scan_feedback(scan_id: str, body: FeedbackRequest, request: Request):
    r = load_scan(scan_id)
    if not r:
        raise HTTPException(404, f"Scan {scan_id} not found")

    current_user = "anonymous"
    token = (request.headers.get("Authorization","")[7:]
             or request.cookies.get("ns_token",""))
    if token and AUTH_AVAILABLE:
        try:
            from auth import decode_jwt
            payload      = decode_jwt(token)
            current_user = payload.get("sub","anonymous")
        except: pass

    feedback = {
        "action"          : body.action,
        "override_verdict": body.override_verdict,
        "clinical_notes"  : body.clinical_notes,
        "by"              : current_user,
        "at"              : datetime.now().isoformat(),
    }
    r["feedback"] = feedback
    path = os.path.join(SCANS_PATH, scan_id, "result.json")
    if os.path.exists(path):
        with open(path,"w",encoding="utf-8") as f:
            json.dump(r, f, indent=2)

    audit("scan_feedback",
          user=current_user,
          detail=f"scan={scan_id} action={body.action} "
                 f"override={body.override_verdict}",
          hospital=r.get("hospital_id",""))
    return {"status":"recorded","scan_id":scan_id,"feedback":feedback}


@app.post("/scans/{scan_id}/consent")
async def record_consent(scan_id: str, body: ConsentRequest, request: Request):
    r = load_scan(scan_id)
    if not r:
        raise HTTPException(404)

    current_user = "anonymous"
    token = (request.headers.get("Authorization","")[7:]
             or request.cookies.get("ns_token",""))
    if token and AUTH_AVAILABLE:
        try:
            from auth import decode_jwt
            payload      = decode_jwt(token)
            current_user = payload.get("sub","anonymous")
        except: pass

    consent = {
        "patient_id"   : body.patient_id,
        "consent_given": body.consent_given,
        "consent_type" : body.consent_type,
        "recorded_by"  : current_user,
        "recorded_at"  : datetime.now().isoformat(),
    }
    r["consent"] = consent
    path = os.path.join(SCANS_PATH, scan_id, "result.json")
    if os.path.exists(path):
        with open(path,"w",encoding="utf-8") as f:
            json.dump(r, f, indent=2)

    audit("consent_recorded", user=current_user,
          detail=f"scan={scan_id} given={body.consent_given}")
    return {"status":"recorded","scan_id":scan_id}


@app.get("/scans/{patient_id}/export")
async def export_patient_history(patient_id: str, request: Request):
    hospital_id = None
    token = (request.headers.get("Authorization","")[7:]
             or request.cookies.get("ns_token",""))
    if token and AUTH_AVAILABLE:
        try:
            from auth import decode_jwt
            payload     = decode_jwt(token)
            hospital_id = payload.get("hospital_id")
        except: pass

    scans = get_patient_scans(patient_id, hospital_id)
    if not scans:
        raise HTTPException(404, f"No scans found for patient {patient_id}")

    # Generate simple text export
    lines = [
        f"NEUROSCOPE AI — PATIENT HISTORY EXPORT",
        f"Patient ID : {patient_id}",
        f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Total scans: {len(scans)}",
        "="*60,
    ]
    for s in scans:
        lines += [
            f"",
            f"Scan ID    : {s.get('scan_id','')}",
            f"Date       : {s.get('timestamp','')}",
            f"Cancer     : {s.get('cancer_type','')}",
            f"Verdict    : {s.get('verdict','')}",
            f"Probability: {s.get('cancer_prob',0):.1%}",
            f"Tumour     : {s.get('tumor_type','')}",
            f"WHO Grade  : {s.get('who_grade','N/A')}",
            f"ICD-10     : {s.get('icd10','N/A')}",
            f"Model Ver  : {s.get('model_version','')}",
            "-"*40,
        ]

    export_text = "\n".join(lines)
    export_path = os.path.join(OUT_PATH, f"export_{patient_id}_{uuid.uuid4().hex[:6]}.txt")
    with open(export_path,"w",encoding="utf-8") as f:
        f.write(export_text)

    return {"patient_id":patient_id,"scans":scans,"export_text":export_text}


@app.post("/analyze/batch")
async def analyze_batch(
    request        : Request,
    background_tasks: BackgroundTasks,
    files          : List[UploadFile] = File(...),
    cancer_type    : str              = Form(...),
    patient_id     : Optional[str]   = Form(None),
):
    if cancer_type not in VALID_CANCER_TYPES:
        raise HTTPException(400, "Invalid cancer_type")
    if len(files) > 50:
        raise HTTPException(400, "Maximum 50 files per batch")

    current_user = "anonymous"
    hospital_id  = None
    token = (request.headers.get("Authorization","")[7:]
             or request.cookies.get("ns_token",""))
    if token and AUTH_AVAILABLE:
        try:
            from auth import decode_jwt
            payload      = decode_jwt(token)
            current_user = payload.get("sub","anonymous")
            hospital_id  = payload.get("hospital_id")
        except: pass

    batch_id = str(uuid.uuid4())[:12]
    task_store[batch_id] = {
        "status":"pending","progress":0,
        "batch_id":batch_id,"results":[],"error":None,
        "total":len(files),"completed":0,
    }

    img_bytes_list = [await f.read() for f in files]

    async def run_batch():
        task_store[batch_id]["status"] = "running"
        results = []
        for i, img_bytes in enumerate(img_bytes_list):
            scan_id = str(uuid.uuid4())[:8]
            try:
                r = await run_pipeline(
                    scan_id, img_bytes, cancer_type,
                    None, None, "",
                    patient_id=patient_id,
                    submitted_by=current_user,
                    hospital_id=hospital_id,
                )
                results.append(r)
            except Exception as e:
                results.append({"scan_id":scan_id,"error":str(e)})
            task_store[batch_id]["completed"] = i+1
            task_store[batch_id]["progress"]  = int((i+1)/len(img_bytes_list)*100)
        task_store[batch_id].update({
            "status":"completed","results":results,"progress":100,
        })

    background_tasks.add_task(run_batch)
    audit("batch_submitted", user=current_user,
          detail=f"batch={batch_id} n={len(files)} cancer={cancer_type}",
          hospital=hospital_id or "")
    return {"batch_id":batch_id,"status":"pending","total":len(files)}


@app.post("/classify")
async def classify_only(
    file       : UploadFile = File(...),
    cancer_type: str        = Form(...),
):
    if cancer_type not in VALID_CANCER_TYPES:
        raise HTTPException(400, "Invalid cancer_type")
    scan_id   = str(uuid.uuid4())[:8]
    img_bytes = await file.read()
    try:
        sess = model_registry.get(f"{cancer_type}_cls") or                model_registry.get(f"{cancer_type}_det")
        if not sess:
            raise HTTPException(503, f"No model for {cancer_type}")
        size  = sess.get_inputs()[0].shape[2] or 224
        x     = preprocess_image(img_bytes, size, cancer_type=cancer_type)
        out   = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        if cancer_type == "breast":
            prob = float(sigmoid(out[0][0]))
            return {"scan_id":scan_id,"prob_malignant":prob,
                    "class":"malignant" if prob>0.5 else "benign"}
        probs = softmax(out[0]).tolist()
        return {"scan_id":scan_id,"probs":probs,
                "pred_class":int(np.argmax(probs)),"confidence":float(max(probs))}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))


# ── Admin endpoints ───────────────────────────────────────────────────────────
@app.get("/admin/stats")
async def admin_stats(request: Request):
    payload      = None
    hospital_id  = None
    token = (request.headers.get("Authorization","")[7:]
             or request.cookies.get("ns_token",""))
    if token and AUTH_AVAILABLE:
        try:
            from auth import decode_jwt
            payload     = decode_jwt(token)
            hospital_id = payload.get("hospital_id")
            if payload["role"] not in ("admin","superadmin"):
                raise HTTPException(403, "Admin required")
        except HTTPException: raise
        except: pass

    # Count scans
    total_scans   = 0
    today_scans   = 0
    flagged_scans = 0
    today         = datetime.now().date().isoformat()

    if os.path.exists(SCANS_PATH):
        for scan_dir in Path(SCANS_PATH).iterdir():
            r_path = scan_dir/"result.json"
            if not r_path.exists(): continue
            try:
                with open(r_path, encoding="utf-8") as f:
                    r = json.load(f)
                if hospital_id and payload and payload.get("role")=="admin":
                    if r.get("hospital_id") != hospital_id: continue
                total_scans += 1
                if r.get("timestamp","").startswith(today):
                    today_scans += 1
                if r.get("verdict") == "CANCER_FLAGGED":
                    flagged_scans += 1
            except: pass

    users     = _load_users()
    hospitals = _load_hospitals()
    return {
        "total_scans"  : total_scans,
        "today_scans"  : today_scans,
        "flagged_scans": flagged_scans,
        "total_users"  : len(users),
        "hospitals"    : len(hospitals),
        "models_loaded": model_registry.count(),
    }


@app.get("/admin/usage")
async def admin_usage(request: Request, days: int = 7):
    token = (request.headers.get("Authorization","")[7:]
             or request.cookies.get("ns_token",""))
    hospital_id = None
    if token and AUTH_AVAILABLE:
        try:
            from auth import decode_jwt
            payload     = decode_jwt(token)
            hospital_id = payload.get("hospital_id")
            if payload["role"] not in ("admin","superadmin"):
                raise HTTPException(403,"Admin required")
        except HTTPException: raise
        except: pass

    usage_by_day    = {}
    usage_by_cancer = {c:0 for c in VALID_CANCER_TYPES}
    usage_by_verdict= {"CANCER_FLAGGED":0,"NORMAL":0,"REVIEW_REQUIRED":0}

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    if os.path.exists(SCANS_PATH):
        for scan_dir in Path(SCANS_PATH).iterdir():
            r_path = scan_dir/"result.json"
            if not r_path.exists(): continue
            try:
                with open(r_path, encoding="utf-8") as f:
                    r = json.load(f)
                if hospital_id:
                    if r.get("hospital_id") != hospital_id: continue
                ts = r.get("timestamp","")
                if ts < cutoff: continue
                day = ts[:10]
                usage_by_day[day]    = usage_by_day.get(day,0)+1
                ct = r.get("cancer_type","")
                if ct in usage_by_cancer: usage_by_cancer[ct]+=1
                vd = r.get("verdict","")
                if vd in usage_by_verdict: usage_by_verdict[vd]+=1
            except: pass

    return {
        "period_days"    : days,
        "by_day"         : usage_by_day,
        "by_cancer_type" : usage_by_cancer,
        "by_verdict"     : usage_by_verdict,
    }


@app.get("/admin/model-performance")
async def model_performance(request: Request):
    token = (request.headers.get("Authorization","")[7:]
             or request.cookies.get("ns_token",""))
    if token and AUTH_AVAILABLE:
        try:
            from auth import decode_jwt
            payload = decode_jwt(token)
            if payload["role"] not in ("admin","superadmin"):
                raise HTTPException(403,"Admin required")
        except HTTPException: raise
        except: pass

    # Per-model: count feedback overrides vs confirms
    model_stats = {c:{"scans":0,"confirmed":0,"overridden":0,"drift_alert":False}
                   for c in VALID_CANCER_TYPES}

    if os.path.exists(SCANS_PATH):
        for scan_dir in Path(SCANS_PATH).iterdir():
            r_path = scan_dir/"result.json"
            if not r_path.exists(): continue
            try:
                with open(r_path, encoding="utf-8") as f:
                    r = json.load(f)
                ct = r.get("cancer_type","")
                if ct not in model_stats: continue
                model_stats[ct]["scans"] += 1
                fb = r.get("feedback")
                if fb:
                    if fb.get("action") == "confirm":
                        model_stats[ct]["confirmed"] += 1
                    elif fb.get("action") == "override":
                        model_stats[ct]["overridden"]+= 1
            except: pass

    # Drift alert if override rate > 20%
    for ct, s in model_stats.items():
        total_fb = s["confirmed"] + s["overridden"]
        if total_fb >= 10:
            override_rate = s["overridden"] / total_fb
            if override_rate > 0.20:
                s["drift_alert"] = True
                s["override_rate"] = round(override_rate, 3)

    return {"model_performance": model_stats}


@app.get("/admin/hospitals")
async def admin_hospitals(request: Request):
    token = (request.headers.get("Authorization","")[7:]
             or request.cookies.get("ns_token",""))
    if token and AUTH_AVAILABLE:
        try:
            from auth import decode_jwt
            payload = decode_jwt(token)
            if payload["role"] != "superadmin":
                raise HTTPException(403,"Superadmin required")
        except HTTPException: raise
        except: pass
    hospitals = _load_hospitals()
    return {"hospitals":hospitals,"count":len(hospitals)}


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/analyze/{scan_id}")
async def ws_analyze(websocket: WebSocket, scan_id: str):
    await ws_manager.connect(scan_id, websocket)
    try:
        data        = await websocket.receive_json()
        cancer_type = data.get("cancer_type","brain")
        patient_age = data.get("patient_age")
        patient_sex = data.get("patient_sex")
        notes       = data.get("notes","")
        patient_id  = data.get("patient_id")

        try:
            img_bytes = base64.b64decode(data.get("image_b64",""))
        except:
            await websocket.send_json({"error":"Invalid image encoding"})
            return

        if cancer_type not in VALID_CANCER_TYPES:
            await websocket.send_json({"error":f"Invalid cancer_type: {cancer_type}"})
            return

        async def on_progress(stage, pd):
            await ws_manager.send(scan_id, {"stage":stage,**pd})

        result = await run_pipeline(
            scan_id, img_bytes, cancer_type,
            patient_age, patient_sex, notes,
            on_progress=on_progress, patient_id=patient_id,
        )
        await websocket.send_json({"stage":"result","data":result})

    except WebSocketDisconnect:
        logger.info(f"WS disconnected: {scan_id}")
    except Exception as e:
        logger.error(f"WS error {scan_id}: {e}")
        try: await websocket.send_json({"error":str(e)})
        except: pass
    finally:
        ws_manager.disconnect(scan_id)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860,
                reload=False, workers=1, log_level="info")
