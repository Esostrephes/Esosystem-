from __future__ import annotations
import base64
import logging
import os
import uuid
import numpy as np
import cv2
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
from collections import Counter

# FastAPI & Supabase
from fastapi import FastAPI, Depends, File, Form, HTTPException, UploadFile, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import BaseModel, EmailStr
from supabase import create_client, Client
import redis

# Biometrics
from insightface.app import FaceAnalysis
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# EMAIL
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UniAttend")

# ====================== CONFIG ======================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
JWT_SECRET = os.getenv("JWT_SECRET", "change_me_in_production_2026")
ALGORITHM = "HS256"
FACE_MATCH_THRESHOLD = 0.40
CURRENT_PERIOD_ID = os.getenv("CURRENT_PERIOD_ID", "2026-1")
ATTENDANCE_THRESHOLD = 80.0
DANGER_THRESHOLD = 75.0

# EMAIL CONFIG
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/login")

# ====================== EMAIL HELPER ======================
def send_email(to_email: str, subject: str, body: str):
    if not EMAIL_USER or not EMAIL_PASS:
        return
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, to_email, msg.as_string())
        server.quit()
    except Exception as e:
        logger.error(f"Email failed: {e}")

# ====================== BIOMETRIC MODELS ======================
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

MODEL_PATH = "blaze_face_short_range.tflite"
# FIX: correct create_from_options signature — BaseOptions goes inside FaceDetectorOptions
detector = vision.FaceDetector.create_from_options(
    vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
)

def extract_embedding(image_input: Union[bytes, np.ndarray]) -> Optional[np.ndarray]:
    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        img = image_input
    if img is None:
        return None
    faces = face_app.get(img)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def b64_to_embedding(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32)

def embedding_to_b64(emb: np.ndarray) -> str:
    return base64.b64encode(emb.astype(np.float32).tobytes()).decode("ascii")

# ====================== REDIS ======================
_redis: Optional[redis.Redis] = None
def get_redis():
    global _redis
    if _redis is None:
        try:
            _redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            _redis.ping()
            logger.info("Redis connected.")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}. Using in-memory fallback.")
    return _redis

# ====================== AUTH ======================
def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated")
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token")
        profile = supabase.table("profiles").select("username, role").eq("id", user_id).single().execute().data
        return {
            "id": user_id,
            "role": profile.get("role"),
            "username": profile.get("username"),
            "email": payload.get("email")
        }
    except JWTError:
        raise HTTPException(401, "Invalid or expired token")
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(401, "Invalid session")

def require_admin(user=Depends(get_current_user)):
    if user["role"] not in ["admin", "super_admin", "hod"]:
        raise HTTPException(403, "Admin/HOD required")
    return user

def require_super_admin(user=Depends(get_current_user)):
    if user["role"] != "super_admin":
        raise HTTPException(403, "Super admin access only")
    return user

# ====================== TIMETABLE HELPERS ======================
def _get_units_for_year(year: int, period_id: str = CURRENT_PERIOD_ID):
    res = supabase.table("timetable") \
        .select("unit_id, sessions_held, units(year_of_study)") \
        .eq("period_id", period_id) \
        .execute()
    return [
        {"unit_id": r["unit_id"], "sessions_held": r["sessions_held"]}
        for r in (res.data or [])
        if r.get("units", {}).get("year_of_study") == year
    ]

def _get_student_year(student_id: str):
    res = supabase.table("students").select("current_year_of_study").eq("id", student_id).single().execute()
    return res.data.get("current_year_of_study") if res.data else None

def _get_students_by_year(year: int):
    res = supabase.table("students") \
        .select("id, name, embedding") \
        .eq("current_year_of_study", year) \
        .eq("is_active", True) \
        .execute()
    return res.data or []

# ====================== APP ======================
app = FastAPI(title="UniAttend v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== LOGIN ======================
# FIX: use Pydantic model so FastAPI can parse the JSON body properly
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@app.post("/admin/login")
def admin_login(payload: LoginRequest):
    try:
        resp = supabase.auth.sign_in_with_password({"email": payload.email, "password": payload.password})
        if not resp.session:
            raise ValueError("No session returned")
        return {
            "access_token": resp.session.access_token,
            "refresh_token": resp.session.refresh_token,
            "token_type": "bearer",
            "expires_in": resp.session.expires_in
        }
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(401, "Invalid email or password")

@app.post("/admin/verify-token")
def verify_token(current_user=Depends(require_admin)):
    return {
        "valid": True,
        "user_id": current_user["id"],
        "username": current_user["username"],
        "role": current_user["role"]
    }

# ====================== UNIVERSITIES / DEPTS / UNITS ======================
@app.get("/universities")
def get_universities():
    result = supabase.table("universities").select("id, name").order("name").execute()
    return result.data or []

@app.get("/departments/{university_id}")
def get_departments(university_id: str):
    result = supabase.table("departments").select("id, name").eq("university_id", university_id).order("name").execute()
    return result.data or []

@app.get("/units/{department_id}")
def get_units(department_id: str):
    rows = supabase.table("units").select("id, code, name, semester").eq("department_id", department_id).eq("is_active", True).order("code").execute().data or []
    for row in rows:
        row["name"] = f"{row['code']} — {row['name']}"
    return rows

# ====================== SESSIONS ======================
class SessionCreate(BaseModel):
    unit_id: str
    date: str
    start_time: Optional[str] = None
    room: Optional[str] = None

@app.get("/sessions/{unit_id}")
def get_sessions(unit_id: str):
    result = supabase.table("sessions").select("id, unit_id, date, start_time, room").eq("unit_id", unit_id).order("date", desc=True).limit(20).execute()
    return result.data or []

@app.post("/sessions")
def create_session(payload: SessionCreate, admin=Depends(require_admin)):
    unit = supabase.table("units").select("id").eq("id", payload.unit_id).maybe_single().execute()
    if not unit.data:
        raise HTTPException(404, "Unit not found")
    existing = supabase.table("sessions").select("id").eq("unit_id", payload.unit_id).eq("date", payload.date).maybe_single().execute()
    if existing.data:
        raise HTTPException(400, "A session already exists for this unit on this date")
    try:
        result = supabase.table("sessions").insert({
            "unit_id": payload.unit_id,
            "date": payload.date,
            "start_time": payload.start_time,
            "room": payload.room,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        raise HTTPException(500, f"Failed to create session: {e}")
    return {"status": "session_created", "session": result.data[0] if result.data else {}}

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, admin=Depends(require_admin)):
    session = supabase.table("sessions").select("id").eq("id", session_id).maybe_single().execute()
    if not session.data:
        raise HTTPException(404, "Session not found")
    supabase.table("sessions").delete().eq("id", session_id).execute()
    return {"status": "session_deleted", "session_id": session_id}

# ====================== ENROLL ======================
@app.post("/enroll")
async def enroll(
    student_id: str = Form(...),
    full_name: str = Form(...),
    email: str = Form(...),
    program: str = Form(""),
    year: int = Form(1),
    phone: str = Form(""),
    university_id: str = Form(...),
    department_id: str = Form(...),
    unit_id: str = Form(...),
    face_image: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    sid = student_id.strip()

    # Verify student exists
    result = supabase.table("students").select("student_id").eq("student_id", sid).maybe_single().execute()
    if not result.data:
        raise HTTPException(404, "Student ID not found. Please register your ID first.")

    img_bytes = await face_image.read()
    embedding = extract_embedding(img_bytes)
    if embedding is None:
        raise HTTPException(422, "No face detected in the uploaded image. Please use a clear, well-lit frontal photo.")

    emb_b64 = embedding_to_b64(embedding)
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        supabase.table("students").update({
            "full_name": full_name.strip(),
            "email": email.strip(),
            "program": program.strip(),
            "year": year,
            "phone": phone.strip(),
            "university_id": university_id,
            "department_id": department_id,
            "unit_id": unit_id,
            "embedding": emb_b64,
            "enrolled_at": now_iso,
        }).eq("student_id", sid).execute()
    except Exception as e:
        raise HTTPException(500, f"Failed to save enrolment: {e}")

    logger.info(f"Enrolled {sid} ({full_name}) — embedding dim={len(embedding)}")

    if background_tasks:
        background_tasks.add_task(
            send_email,
            email.strip(),
            "UniAttend Enrollment Successful",
            f"Hi {full_name},\n\nYour face has been registered successfully!\nStudent ID: {sid}\nYou can now attend classes with facial recognition.\n\nUniAttend v3"
        )

    return {"status": "enrolled", "student_id": sid, "full_name": full_name.strip(), "embedding_dim": int(len(embedding))}

# ====================== STUDENTS LIST ======================
@app.get("/students")
def list_students(admin=Depends(require_admin)):
    result = supabase.table("students").select(
        "student_id, full_name, email, program, year, phone, enrolled_at, created_at, embedding"
    ).order("created_at", desc=True).execute()
    rows = result.data or []
    for row in rows:
        row["has_face"] = row.pop("embedding") is not None
    return rows

# ====================== BATCH ATTENDANCE ======================
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB

@app.post("/attendance/batch")
async def batch_attendance(
    session_id: str = Form(...),
    group_photo: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    current_user=Depends(require_admin),
):
    session_id = session_id.strip()
    if not session_id:
        raise HTTPException(422, "session_id is required")
    if group_photo.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"Unsupported image type: {group_photo.content_type}")

    img_bytes = await group_photo.read()
    if len(img_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds 10 MB limit")

    # Load session → unit → department → students (original lookup chain)
    try:
        session_row = supabase.table("sessions").select("unit_id").eq("id", session_id).maybe_single().execute()
        if not session_row.data:
            raise HTTPException(404, f"Session '{session_id}' not found")
        unit_id = session_row.data["unit_id"]

        unit_row = supabase.table("units").select("department_id").eq("id", unit_id).maybe_single().execute()
        if not unit_row.data:
            raise HTTPException(404, f"Unit '{unit_id}' not found")
        dept_id = unit_row.data["department_id"]

        students_res = supabase.table("students").select("student_id, embedding, email, full_name") \
            .eq("department_id", dept_id).not_.is_("embedding", "null").execute()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("DB error fetching students")
        raise HTTPException(502, "Database error fetching students") from exc

    students = students_res.data or []
    if not students:
        raise HTTPException(404, "No students with embeddings in this department")

    enrolled: Dict[str, np.ndarray] = {
        s["student_id"]: b64_to_embedding(s["embedding"])
        for s in students if s.get("embedding")
    }

    # Detect faces
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "Could not decode image")

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = detector.detect(mp_image)
    detected_faces = detection_result.detections or []

    if not detected_faces:
        raise HTTPException(422, "No faces detected in the group photo")

    attendance_rows: List[dict] = []
    unmatched_rows: List[dict] = []
    response_faces: List[dict] = []
    already_marked: set[str] = set()   # FIX: dedup — same student can't be marked twice
    now_iso = datetime.now(timezone.utc).isoformat()

    for detection in detected_faces:
        bbox = detection.bounding_box

        # FIX: 20% padding restored (same as original) to avoid tight crops failing InsightFace
        pad_x = int(bbox.width * 0.2)
        pad_y = int(bbox.height * 0.2)
        x1 = max(0, int(bbox.origin_x) - pad_x)
        y1 = max(0, int(bbox.origin_y) - pad_y)
        x2 = min(w, int(bbox.origin_x + bbox.width) + pad_x)
        y2 = min(h, int(bbox.origin_y + bbox.height) + pad_y)
        crop = img[y1:y2, x1:x2]

        live_emb = extract_embedding(crop)
        if live_emb is None:
            logger.warning("Embedding extraction failed for a detected face; skipping.")
            continue

        if not enrolled:
            continue

        best_id = max(enrolled, key=lambda sid: cosine_similarity(live_emb, enrolled[sid]))
        best_score = cosine_similarity(live_emb, enrolled[best_id])

        box = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}

        # Duplicate within this request
        if best_score >= FACE_MATCH_THRESHOLD and best_id in already_marked:
            response_faces.append({"student_id": best_id, "score": round(best_score, 4), "box": box, "status": "duplicate"})
            continue

        if best_score >= FACE_MATCH_THRESHOLD:
            already_marked.add(best_id)
            attendance_rows.append({
                "student_id": best_id,
                "session_id": session_id,
                "period_id": CURRENT_PERIOD_ID,
                "timestamp": now_iso,
                "face_score": round(best_score, 6),
                "face_verified": True,
            })
            response_faces.append({"student_id": best_id, "score": round(best_score, 4), "box": box, "status": "verified"})

            # Email student on successful attendance
            student = next((s for s in students if s["student_id"] == best_id), None)
            if student and student.get("email") and background_tasks:
                background_tasks.add_task(
                    send_email,
                    student["email"],
                    "✅ Attendance Marked",
                    f"Hi {student['full_name']},\n\nYou have been marked PRESENT.\nSession: {session_id}\nTime: {now_iso}\n\nUniAttend v3"
                )
        else:
            try:
                _, buf = cv2.imencode(".jpg", crop)
                path = f"unmatched/{session_id}/{uuid.uuid4()}.jpg"
                supabase.storage.from_("face-crops").upload(path, buf.tobytes())
                crop_url = supabase.storage.from_("face-crops").get_public_url(path)
            except Exception:
                logger.exception("Failed to upload unmatched crop; storing without URL.")
                crop_url = None

            unmatched_rows.append({"session_id": session_id, "timestamp": now_iso, "crop_url": crop_url, "reviewed": False})
            response_faces.append({"student_id": None, "score": round(best_score, 4), "box": box, "status": "unrecognized"})

    # Bulk upsert attendance
    if attendance_rows:
        try:
            supabase.table("attendance").upsert(
                attendance_rows,
                on_conflict="student_id,session_id"
            ).execute()
        except Exception as exc:
            logger.exception("DB error inserting attendance")
            raise HTTPException(502, "Failed to record attendance") from exc

    # Store unmatched (non-fatal)
    if unmatched_rows:
        try:
            supabase.table("unmatched_faces").insert(unmatched_rows).execute()
        except Exception:
            logger.exception("Failed to store unmatched face records")

    return {
        "status": "batch_processed",
        "faces_detected": len(detected_faces),
        "students_matched": len(attendance_rows),
        "unmatched_faces": len(unmatched_rows),
        "faces": response_faces,
    }

# ====================== TIMETABLE ======================
@app.post("/report-session")
async def report_session(unit_id: str, status: str, period_id: str = CURRENT_PERIOD_ID, current_user=Depends(require_admin)):
    unit_res = supabase.table("timetable").select("sessions_held") \
        .eq("unit_id", unit_id).eq("period_id", period_id).single().execute()
    if not unit_res.data:
        raise HTTPException(404, "Unit not found in timetable for this period")
    if status.upper() == "HELD":
        new_count = (unit_res.data.get("sessions_held") or 0) + 1
        # FIX: filter by period_id so you don't update all periods
        supabase.table("timetable").update({"sessions_held": new_count}) \
            .eq("unit_id", unit_id).eq("period_id", period_id).execute()
    supabase.table("session_log").insert({
        "unit_id": unit_id,
        "status": status.upper(),
        "reported_by": current_user["id"],
        "period_id": period_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }).execute()
    return {"message": f"Unit {unit_id} marked as {status.upper()} for {period_id}"}

# ====================== ANALYTICS ======================
@app.get("/analytics/defaulters/{year}")
async def get_defaulters(year: int, current_user=Depends(require_admin)):
    units = _get_units_for_year(year)
    students = _get_students_by_year(year)
    s_ids = [s["id"] for s in students]
    if not s_ids:
        return []

    logs = supabase.table("attendance").select("student_id, unit_id") \
        .eq("period_id", CURRENT_PERIOD_ID).in_("student_id", s_ids).execute().data or []
    attendance_counts = Counter((l["student_id"], l["unit_id"]) for l in logs)

    defaulter_list = []
    for s in students:
        for tt in units:
            u_id = tt["unit_id"]
            held = tt["sessions_held"] or 0
            attended = attendance_counts.get((s["id"], u_id), 0)
            percentage = round((attended / held * 100), 2) if held > 0 else 100.0
            if percentage < ATTENDANCE_THRESHOLD:
                defaulter_list.append({
                    "name": s["name"],
                    "unit": u_id,
                    "percentage": f"{percentage}%",
                    "status": "DANGER" if percentage < DANGER_THRESHOLD else "WARNING",
                    "exam_eligible": False
                })
    return sorted(defaulter_list, key=lambda x: x["name"])

@app.get("/analytics/student-report/{student_id}")
async def get_student_report(student_id: str, current_user=Depends(get_current_user)):
    if current_user["role"] == "student" and current_user["id"] != student_id:
        raise HTTPException(403, "Access denied")
    student = supabase.table("students").select("id, is_active").eq("id", student_id).single().execute().data
    if not student or not student.get("is_active"):
        raise HTTPException(404, "Student not found or deactivated")
    year = _get_student_year(student_id)
    if not year:
        raise HTTPException(400, "Student year not set")
    units = _get_units_for_year(year)
    logs = supabase.table("attendance").select("unit_id").eq("student_id", student_id).eq("period_id", CURRENT_PERIOD_ID).execute().data or []
    my_counts = Counter(l["unit_id"] for l in logs)
    breakdown = []
    total_held = total_attended = 0
    for u in units:
        held = u["sessions_held"] or 0
        attended = my_counts.get(u["unit_id"], 0)
        perc = round((attended / held * 100), 2) if held > 0 else 100.0
        breakdown.append({
            "unit": u["unit_id"],
            "attendance": f"{perc}%",
            "eligible": perc >= ATTENDANCE_THRESHOLD,
            "color_code": "RED" if perc < ATTENDANCE_THRESHOLD else "GREEN"
        })
        total_held += held
        total_attended += attended
    overall = round((total_attended / total_held * 100), 2) if total_held > 0 else 100.0
    return {
        "breakdown": breakdown,
        "overall_attendance": f"{overall}%",
        "overall_exam_eligible": overall >= ATTENDANCE_THRESHOLD,
        "period": CURRENT_PERIOD_ID
    }

@app.get("/analytics/department-census")
async def get_department_census(current_user=Depends(require_admin)):
    students = supabase.table("students").select("id, embedding, current_year_of_study").eq("is_active", True).execute().data or []
    today = datetime.now(timezone.utc).date().isoformat()
    today_logs = supabase.table("attendance").select("student_id").eq("period_id", CURRENT_PERIOD_ID).gte("timestamp", today).execute().data or []
    active_today = len({log["student_id"] for log in today_logs})
    stats_by_year = {y: {"reg": 0, "bio": 0} for y in range(1, 5)}
    for s in students:
        y = s.get("current_year_of_study")
        if y in stats_by_year:
            stats_by_year[y]["reg"] += 1
            if s.get("embedding"):
                stats_by_year[y]["bio"] += 1
    return {
        "summary": {"total_students": len(students), "active_today": active_today, "period": CURRENT_PERIOD_ID},
        "chart_arrays": {
            "labels": ["Year 1", "Year 2", "Year 3", "Year 4"],
            "registered": [stats_by_year[y]["reg"] for y in range(1, 5)],
            "biometric_enrolled": [stats_by_year[y]["bio"] for y in range(1, 5)],
        }
    }

# ====================== SUPER ADMIN ======================
@app.post("/analytics/admin/manual-attendance")
async def manual_attendance(student_id: str, unit_id: str, is_excused: bool = False, period_id: str = CURRENT_PERIOD_ID, current_user=Depends(require_super_admin)):
    supabase.table("attendance").insert({
        "student_id": student_id,
        "unit_id": unit_id,
        "period_id": period_id,
        "is_excused": is_excused,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }).execute()
    return {"message": f"Manual {'excused' if is_excused else 'present'} attendance added"}

@app.delete("/analytics/admin/attendance/{attendance_id}")
async def delete_attendance(attendance_id: str, current_user=Depends(require_super_admin)):
    res = supabase.table("attendance").delete().eq("id", attendance_id).execute()
    if not res.data:
        raise HTTPException(404, "Attendance record not found")
    return {"message": "Attendance record deleted"}

@app.post("/analytics/admin/deactivate-student/{student_id}")
async def deactivate_student(student_id: str, reason: Optional[str] = None, current_user=Depends(require_super_admin)):
    update = {"is_active": False}
    if reason:
        update["deactivation_reason"] = reason
    res = supabase.table("students").update(update).eq("id", student_id).execute()
    if not res.data:
        raise HTTPException(404, "Student not found")
    return {"message": f"Student {student_id} deactivated"}

# ====================== FRONTEND ======================
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>UniAttend v3</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,300&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --ink:#0d0e14;--ink2:#3a3b45;--paper:#f4f3ee;--surface:#ffffff;--surface2:#eeeee8;
  --border:#dddcd4;--border2:#c8c7be;
  --accent:#1a3599;--accent-lt:#eaedfa;--accent-mid:#3d5cc4;
  --green:#166740;--green-lt:#e5f3ec;--green-mid:#1d8a52;
  --red:#b83228;--red-lt:#fdeeed;
  --amber:#9a5a00;--amber-lt:#fef4e2;
  --muted:#86857a;--muted2:#b5b4ac;
  --font-h:'Plus Jakarta Sans',sans-serif;
  --font-d:'Fraunces',serif;
  --font-m:'DM Mono',monospace;
  --r:6px;--r-lg:12px;--r-xl:18px;
  --shadow:0 1px 2px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.05);
  --shadow-lg:0 2px 4px rgba(0,0,0,.07),0 8px 32px rgba(0,0,0,.1);
}
body{font-family:var(--font-h);background:var(--paper);color:var(--ink);min-height:100vh}
.app{display:flex;min-height:100vh}
.sidebar{width:232px;background:var(--ink);color:#fff;display:flex;flex-direction:column;position:fixed;top:0;left:0;height:100vh;z-index:100}
.sidebar-brand{padding:22px 18px 18px;border-bottom:1px solid rgba(255,255,255,.07)}
.brand-logo{width:36px;height:36px;background:var(--accent-mid);border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.9rem;margin-bottom:8px}
.brand-title{font-family:var(--font-d);font-size:1.2rem;font-weight:600;letter-spacing:-.01em}
.brand-sub{font-family:var(--font-m);font-size:.55rem;color:rgba(255,255,255,.35);margin-top:3px}
.sidebar-nav{flex:1;padding:14px 8px;overflow-y:auto;scrollbar-width:none}
.nav-section-label{font-family:var(--font-m);font-size:.52rem;color:rgba(255,255,255,.3);text-transform:uppercase;letter-spacing:.08em;padding:10px 10px 4px}
.nav-item{display:flex;align-items:center;gap:10px;padding:9px 10px;border-radius:var(--r);cursor:pointer;border:none;background:transparent;color:rgba(255,255,255,.6);width:100%;text-align:left;font-family:var(--font-h);font-size:.82rem;font-weight:500;transition:.15s}
.nav-item:hover{color:rgba(255,255,255,.9);background:rgba(255,255,255,.06)}
.nav-item.active{color:#fff;background:var(--accent-mid)}
.nav-item .ni{width:18px;text-align:center;flex-shrink:0;font-size:.88rem}
.admin-only{display:none}
.admin-only.visible{display:block}
.sidebar-footer{padding:14px 8px;border-top:1px solid rgba(255,255,255,.07)}
.user-card{display:flex;align-items:center;gap:9px;padding:9px 10px;border-radius:var(--r);background:rgba(255,255,255,.05);margin-bottom:8px}
.user-avatar{width:28px;height:28px;border-radius:50%;background:var(--accent-mid);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.75rem;flex-shrink:0}
.user-name{font-size:.78rem;font-weight:700;color:#fff;line-height:1.2}
.user-role{font-family:var(--font-m);font-size:.55rem;color:rgba(255,255,255,.35)}
.btn-logout{width:100%;padding:7px;border:1px solid rgba(255,255,255,.1);background:transparent;color:rgba(255,255,255,.5);border-radius:var(--r);cursor:pointer;font-family:var(--font-m);font-size:.65rem;transition:.15s}
.btn-logout:hover{border-color:rgba(255,255,255,.2);color:rgba(255,255,255,.75)}
.main{margin-left:232px;flex:1;display:flex;flex-direction:column;min-height:100vh}
.topbar{height:56px;border-bottom:1.5px solid var(--border);background:var(--surface);display:flex;align-items:center;gap:14px;padding:0 28px;position:sticky;top:0;z-index:50}
.page-title{font-family:var(--font-d);font-size:1.1rem;font-weight:600;color:var(--ink);flex:1}
.topbar-chip{font-family:var(--font-m);font-size:.6rem;padding:3px 9px;border-radius:10px;background:var(--accent-lt);color:var(--accent)}
.content{padding:28px;max-width:820px}
.panel{display:none;animation:rise .22s ease}
.panel.active{display:block}
@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.card{background:var(--surface);border:1.5px solid var(--border);border-radius:var(--r-lg);padding:20px;margin-bottom:16px;box-shadow:var(--shadow)}
.card-label{font-family:var(--font-m);font-size:.58rem;font-weight:500;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:14px}
.field{margin-bottom:11px}
.field label{display:block;font-family:var(--font-m);font-size:.58rem;color:var(--muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em}
input,select,textarea{width:100%;background:var(--surface2);border:1.5px solid var(--border);border-radius:var(--r);padding:9px 11px;font-family:var(--font-h);font-size:.85rem;color:var(--ink);outline:none;transition:.15s}
input:focus,select:focus,textarea:focus{border-color:var(--accent);background:#fff}
select{appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2386857a' d='M6 8L1 3h10z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 11px center}
select:disabled{opacity:.45;cursor:not-allowed}
.split{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:600px){.split{grid-template-columns:1fr}}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;padding:9px 18px;border-radius:var(--r);border:none;cursor:pointer;font-family:var(--font-h);font-size:.82rem;font-weight:600;transition:.15s}
.btn:disabled{opacity:.38;cursor:not-allowed}
.btn-primary{background:var(--accent);color:#fff}
.btn-primary:hover:not(:disabled){background:#162e82}
.btn-green{background:var(--green-mid);color:#fff}
.btn-green:hover:not(:disabled){background:var(--green)}
.btn-ghost{background:transparent;border:1.5px solid var(--border);color:var(--ink2)}
.btn-ghost:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}
.btn-danger{background:transparent;border:1.5px solid #f0c4c0;color:var(--red)}
.btn-danger:hover:not(:disabled){background:var(--red-lt)}
.btn-full{width:100%}
.status{display:none;align-items:center;gap:9px;padding:10px 13px;border-radius:var(--r);font-family:var(--font-m);font-size:.72rem;margin-bottom:10px}
.status.show{display:flex;animation:rise .2s ease}
.status.ok{background:var(--green-lt);border:1.5px solid #b7dfc8;color:var(--green)}
.status.err{background:var(--red-lt);border:1.5px solid #f5c6c2;color:var(--red)}
.status.info{background:var(--accent-lt);border:1.5px solid #c3cfed;color:var(--accent)}
.status.warn{background:var(--amber-lt);border:1.5px solid #fcd48f;color:var(--amber)}
.prog{height:3px;background:var(--border);border-radius:2px;overflow:hidden;margin:10px 0}
.prog-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--green-mid));width:0%;transition:width .4s ease}
.steps{display:flex;gap:0;margin-bottom:22px;position:relative}
.steps::before{content:'';position:absolute;top:15px;left:16px;right:16px;height:1.5px;background:var(--border);z-index:0}
.step{flex:1;display:flex;flex-direction:column;align-items:center;gap:5px;position:relative;z-index:1}
.step-dot{width:30px;height:30px;border-radius:50%;background:var(--surface2);border:2px solid var(--border);display:flex;align-items:center;justify-content:center;font-family:var(--font-m);font-size:.7rem;font-weight:500;color:var(--muted2);transition:.2s}
.step.done .step-dot{background:var(--green-mid);border-color:var(--green-mid);color:#fff}
.step.active .step-dot{background:var(--accent);border-color:var(--accent);color:#fff}
.step-lbl{font-family:var(--font-m);font-size:.56rem;color:var(--muted2);text-transform:uppercase;letter-spacing:.05em;text-align:center}
.step.active .step-lbl{color:var(--accent)}
.step.done .step-lbl{color:var(--green)}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px}
.stat{background:var(--surface2);border:1.5px solid var(--border);border-radius:var(--r);padding:14px}
.stat-val{font-family:var(--font-d);font-size:1.75rem;font-weight:600;color:var(--ink)}
.stat-lbl{font-family:var(--font-m);font-size:.58rem;color:var(--muted);text-transform:uppercase;margin-top:2px}
.tbl-wrap{border:1.5px solid var(--border);border-radius:var(--r);overflow:auto;margin-top:10px}
table{width:100%;border-collapse:collapse}
th{font-family:var(--font-m);font-size:.57rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;padding:9px 12px;border-bottom:1.5px solid var(--border);text-align:left;background:var(--surface2)}
td{font-family:var(--font-m);font-size:.76rem;padding:9px 12px;border-bottom:1.5px solid var(--border)}
tr:last-child td{border:none}
tr:hover td{background:var(--surface2)}
.badge{display:inline-block;font-size:.6rem;padding:2px 7px;border-radius:10px}
.b-yes{background:var(--green-lt);color:var(--green)}
.b-no{background:var(--amber-lt);color:var(--amber)}
.b-fail{background:var(--red-lt);color:var(--red)}
.drop-zone{border:2px dashed var(--border);border-radius:var(--r-lg);padding:28px 20px;text-align:center;cursor:pointer;transition:.2s;margin-bottom:12px}
.drop-zone:hover,.drop-zone.drag{border-color:var(--accent);background:var(--accent-lt)}
.drop-zone input{display:none}
.drop-zone-icon{font-size:1.8rem;margin-bottom:7px}
.drop-zone-label{font-family:var(--font-m);font-size:.75rem;color:var(--muted)}
.drop-zone-label span{color:var(--accent);font-weight:500}
.face-preview{width:100%;max-width:200px;aspect-ratio:1;object-fit:cover;border-radius:var(--r);border:2px solid var(--border);display:none}
.face-preview.show{display:block}
.photo-tabs{display:flex;gap:2px;background:var(--surface2);border:1.5px solid var(--border);border-radius:var(--r);padding:3px;margin-bottom:12px}
.photo-tab{flex:1;padding:8px;border:none;background:transparent;font-family:var(--font-m);font-size:.68rem;cursor:pointer;border-radius:4px;color:var(--muted);transition:.15s}
.photo-tab.active{background:var(--surface);color:var(--accent);box-shadow:0 1px 3px rgba(0,0,0,.08)}
.cam-wrap{position:relative;width:100%;aspect-ratio:4/3;background:#0d0e14;border-radius:var(--r);overflow:hidden;margin-bottom:8px}
.cam-overlay{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;font-family:var(--font-m);font-size:.72rem;color:rgba(255,255,255,.4)}
.cam-ring{width:100px;height:100px;border-radius:50%;border:2px dashed rgba(255,255,255,.2)}
.cam-snap-preview{width:100%;max-width:180px;border-radius:var(--r);border:2px solid var(--green-mid);display:none;margin-top:8px}
.consent-body{background:var(--surface2);border:1.5px solid var(--border);border-radius:var(--r);padding:14px;margin-bottom:14px;max-height:200px;overflow-y:auto}
.consent-body p{font-family:var(--font-m);font-size:.72rem;color:var(--ink2);line-height:1.6;margin-bottom:10px}
.consent-check{display:flex;align-items:flex-start;gap:10px;margin-bottom:14px}
.consent-check input[type=checkbox]{width:16px;height:16px;flex-shrink:0;margin-top:2px}
.consent-check-label{font-family:var(--font-m);font-size:.73rem;color:var(--ink2);line-height:1.5}
.divider{height:1.5px;background:var(--border);margin:16px 0}
.blist{border:1.5px solid var(--border);border-radius:var(--r);max-height:300px;overflow-y:auto}
.bi{display:flex;justify-content:space-between;align-items:center;padding:9px 12px;border-bottom:1.5px solid var(--border);font-family:var(--font-m);font-size:.75rem}
.bi:last-child{border:none}
.pill{font-size:.62rem;padding:2px 8px;border-radius:10px}
.pill-ok{background:var(--green-lt);color:var(--green)}
.pill-fail{background:var(--red-lt);color:var(--red)}
.pill-warn{background:var(--amber-lt);color:var(--amber)}
.spin{width:14px;height:14px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spn .7s linear infinite;flex-shrink:0}
@keyframes spn{to{transform:rotate(360deg)}}
.sess-row{display:flex;justify-content:space-between;align-items:center;padding:10px 12px;border-bottom:1.5px solid var(--border)}
.sess-row:last-child{border-bottom:none}
.sess-meta{font-size:.65rem;color:var(--muted);margin-top:2px}
@media(max-width:768px){
  .sidebar{transform:translateX(-100%);transition:.3s}
  .sidebar.open{transform:none}
  .main{margin-left:0}
  .mob-menu-btn{display:flex}
  .content{padding:18px}
}
.mob-menu-btn{display:none;align-items:center;justify-content:center;width:36px;height:36px;border:none;background:transparent;cursor:pointer;font-size:1.2rem}
.mob-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:99}
.mob-overlay.show{display:block}
</style>
</head>
<body>
<div class="app">
  <nav class="sidebar" id="sidebar">
    <div class="sidebar-brand">
      <div class="brand-logo">UA</div>
      <div class="brand-title">UniAttend</div>
      <div class="brand-sub">Facial Attendance · v3.0</div>
    </div>
    <div class="sidebar-nav">
      <div class="nav-section-label">Student</div>
      <button class="nav-item active" onclick="showPanel('enrol')" id="nav-enrol">
        <span class="ni">🎓</span><span>Enrolment</span>
      </button>
      <div class="admin-only" id="admin-nav-section">
        <div class="nav-section-label">Admin</div>
        <button class="nav-item" onclick="showPanel('batch')" id="nav-batch">
          <span class="ni">📷</span><span>Batch Attendance</span>
        </button>
        <button class="nav-item" onclick="showPanel('sessions')" id="nav-sessions">
          <span class="ni">📅</span><span>Sessions</span>
        </button>
        <button class="nav-item" onclick="showPanel('students')" id="nav-students">
          <span class="ni">👥</span><span>Students</span>
        </button>
        <button class="nav-item" onclick="showPanel('analytics')" id="nav-analytics">
          <span class="ni">📊</span><span>Analytics</span>
        </button>
      </div>
      <div class="nav-section-label" style="margin-top:10px">Account</div>
      <button class="nav-item" onclick="showPanel('admin')" id="nav-admin">
        <span class="ni">🔐</span><span id="nav-admin-label">Admin Login</span>
      </button>
    </div>
    <div class="sidebar-footer">
      <div id="sidebar-user" style="display:none">
        <div class="user-card">
          <div class="user-avatar" id="user-avatar-initial">A</div>
          <div>
            <div class="user-name" id="user-display-name">Admin</div>
            <div class="user-role" id="user-display-role">admin</div>
          </div>
        </div>
        <button class="btn-logout" onclick="doLogout()">Sign out</button>
      </div>
      <div id="sidebar-guest" style="font-family:var(--font-m);font-size:.65rem;color:rgba(255,255,255,.3);padding:8px 10px">Not signed in</div>
    </div>
  </nav>
  <div class="mob-overlay" id="mob-overlay" onclick="closeSidebar()"></div>
  <div class="main">
    <div class="topbar">
      <button class="mob-menu-btn" onclick="openSidebar()">☰</button>
      <div class="page-title" id="page-title">Student Enrolment</div>
      <span class="topbar-chip">UniAttend v3</span>
    </div>
    <div class="content">

      <!-- ENROLMENT PANEL -->
      <div class="panel active" id="panel-enrol">
        <div class="steps">
          <div class="step active" id="st0"><div class="step-dot">1</div><div class="step-lbl">Details</div></div>
          <div class="step" id="st1"><div class="step-dot">2</div><div class="step-lbl">Unit</div></div>
          <div class="step" id="st2"><div class="step-dot">3</div><div class="step-lbl">Consent</div></div>
          <div class="step" id="st3"><div class="step-dot">4</div><div class="step-lbl">Face</div></div>
          <div class="step" id="st4"><div class="step-dot">5</div><div class="step-lbl">Done</div></div>
        </div>
        <div id="enrol-step0" class="card">
          <div class="card-label">Personal Details</div>
          <div class="split">
            <div>
              <div class="field"><label>Student ID *</label><input id="e-sid" placeholder="e.g. SCT221-0001/2022"></div>
              <div class="field"><label>Full Name *</label><input id="e-name" placeholder="As on admission letter"></div>
              <div class="field"><label>Email</label><input id="e-email" type="email" placeholder="student@university.ac.ke"></div>
              <div class="field"><label>Phone</label><input id="e-phone" type="tel" placeholder="+254..."></div>
            </div>
            <div>
              <div class="field"><label>Program</label><input id="e-prog" placeholder="e.g. BSc Computer Science"></div>
              <div class="field"><label>Year of Study</label>
                <select id="e-year">
                  <option value="1">Year 1</option><option value="2">Year 2</option>
                  <option value="3">Year 3</option><option value="4">Year 4</option>
                  <option value="5">Year 5</option><option value="6">Year 6</option>
                </select>
              </div>
            </div>
          </div>
          <div id="st-reg" class="status"></div>
          <button class="btn btn-primary btn-full" onclick="enrolStep1()">Continue →</button>
        </div>
        <div id="enrol-step1" style="display:none">
          <div class="card">
            <div class="card-label">Academic Details</div>
            <div class="field"><label>University *</label><select id="e-uni" onchange="eLoadDepts()"><option value="">— Select University —</option></select></div>
            <div class="field"><label>Department *</label><select id="e-dept" onchange="eLoadUnits()" disabled><option value="">— Select Department —</option></select></div>
            <div class="field"><label>Unit *</label><select id="e-unit" disabled><option value="">— Select Unit —</option></select></div>
            <div id="st-unit" class="status"></div>
            <div style="display:flex;gap:8px;margin-top:4px">
              <button class="btn btn-ghost" onclick="goEnrolStep(0)">← Back</button>
              <button class="btn btn-primary btn-full" onclick="enrolStep2()">Continue →</button>
            </div>
          </div>
        </div>
        <div id="enrol-step2" style="display:none">
          <div class="card">
            <div class="card-label">Biometric Data Consent</div>
            <div class="consent-body">
              <p><strong>What we collect:</strong> A mathematical representation of your facial geometry (512 numbers). No photograph is stored.</p>
              <p><strong>How it is used:</strong> Your facial embedding is used solely to mark your attendance in registered units.</p>
              <p><strong>Storage & security:</strong> Embeddings are stored in Supabase with row-level security. Only authorised admins can access your record.</p>
              <p><strong>Your rights under the Kenya Data Protection Act 2019:</strong> You may request access to, correction of, or deletion of your data at any time via your department office.</p>
              <p><strong>Retention:</strong> Your data is retained for the duration of your enrolment plus one academic year.</p>
            </div>
            <div class="consent-check">
              <input type="checkbox" id="consent-check" onchange="checkConsentReady()">
              <label for="consent-check" class="consent-check-label">I have read and understood the above. I voluntarily consent to the collection and use of my biometric data for attendance purposes.</label>
            </div>
            <div class="field">
              <label>Sign with your full name *</label>
              <input id="consent-signature" placeholder="Type your full name to sign" oninput="checkConsentReady()">
            </div>
            <div id="st-consent" class="status"></div>
            <div style="display:flex;gap:8px;margin-top:4px">
              <button class="btn btn-ghost" onclick="goEnrolStep(1)">← Back</button>
              <button class="btn btn-primary btn-full" id="btn-proceed-face" onclick="enrolStep3()" disabled>Proceed to Face Capture →</button>
            </div>
          </div>
        </div>
        <div id="enrol-step3" style="display:none">
          <div class="card">
            <div class="card-label">Face Photo</div>
            <div class="split">
              <div>
                <div id="e-face-status" class="status show info">Upload or capture a clear frontal photo.</div>
                <img id="e-preview" class="face-preview" alt="Face preview">
                <div style="margin-top:12px;font-family:var(--font-m);font-size:.68rem;color:var(--muted);line-height:1.8">
                  ✦ Remove glasses if possible<br>✦ Neutral expression<br>✦ Face the camera directly<br>✦ Good lighting
                </div>
              </div>
              <div>
                <div class="photo-tabs">
                  <button class="photo-tab active" id="enr-tab-upload" onclick="switchEnrTab('upload')">📁 Upload</button>
                  <button class="photo-tab" id="enr-tab-camera" onclick="switchEnrTab('camera')">📷 Camera</button>
                </div>
                <div id="enr-upload-pane">
                  <div class="drop-zone" id="enr-drop" onclick="document.getElementById('enr-file').click()"
                       ondragover="event.preventDefault();this.classList.add('drag')"
                       ondragleave="this.classList.remove('drag')"
                       ondrop="enrDrop(event)">
                    <input type="file" id="enr-file" accept="image/jpeg,image/png,image/webp" onchange="enrFileSelect(event)">
                    <div class="drop-zone-icon">📷</div>
                    <div class="drop-zone-label">Drop photo here or <span>browse</span></div>
                  </div>
                </div>
                <div id="enr-camera-pane" style="display:none">
                  <div class="cam-wrap" id="enr-cam-wrap">
                    <video id="enr-cam-video" autoplay playsinline muted style="width:100%;height:100%;object-fit:cover;display:none"></video>
                    <canvas id="enr-cam-canvas" style="display:none"></canvas>
                    <div class="cam-overlay" id="enr-cam-overlay"><div class="cam-ring"></div><span>Camera off</span></div>
                  </div>
                  <div style="display:flex;gap:8px;margin-top:10px">
                    <button class="btn btn-ghost btn-full" id="enr-cam-start-btn" onclick="enrStartCamera()">Start Camera</button>
                    <button class="btn btn-primary btn-full" id="enr-cam-snap-btn" onclick="enrSnapPhoto()" disabled>Capture</button>
                  </div>
                  <img id="enr-cam-preview" class="cam-snap-preview" alt="Captured">
                </div>
              </div>
            </div>
            <div class="divider"></div>
            <div id="st-enrol" class="status"></div>
            <div style="display:flex;gap:8px;margin-top:4px">
              <button class="btn btn-ghost" onclick="goEnrolStep(2)">← Back</button>
              <button class="btn btn-green btn-full" id="btn-submit-enrol" onclick="submitEnrolment()" disabled>Submit Enrolment</button>
            </div>
          </div>
        </div>
        <div id="enrol-step4" style="display:none">
          <div class="card" style="text-align:center;padding:44px 20px">
            <div style="font-size:3rem;margin-bottom:14px">🎓</div>
            <div style="font-family:var(--font-d);font-size:1.3rem;font-weight:600;margin-bottom:8px" id="enrol-done-name">Enrolled!</div>
            <div style="font-family:var(--font-m);font-size:.72rem;color:var(--muted);margin-bottom:6px" id="enrol-done-meta"></div>
            <div style="font-family:var(--font-m);font-size:.7rem;color:var(--green)">✅ Facial embedding stored · Attendance-ready · Confirmation email sent</div>
            <button class="btn btn-ghost" style="margin-top:20px" onclick="resetEnrol()">Enrol Another Student</button>
          </div>
        </div>
      </div>

      <!-- BATCH ATTENDANCE PANEL -->
      <div class="panel" id="panel-batch">
        <div class="card">
          <div class="card-label">Batch — Group Photo Attendance</div>
          <div class="split">
            <div class="field"><label>University</label><select id="b-uni" onchange="bLoadDepts()"><option value="">— Select University —</option></select></div>
            <div class="field"><label>Department</label><select id="b-dept" onchange="bLoadUnits()" disabled><option value="">— Select Department —</option></select></div>
          </div>
          <div class="split">
            <div class="field"><label>Unit</label><select id="b-unit" onchange="bLoadSessions()" disabled><option value="">— Select Unit —</option></select></div>
            <div class="field"><label>Session</label><select id="b-session" disabled onchange="checkBatchReady()"><option value="">— Select Session —</option></select></div>
          </div>
          <div class="divider"></div>
          <div class="photo-tabs">
            <button class="photo-tab active" id="batch-tab-upload" onclick="switchBatchTab('upload')">📁 Upload</button>
            <button class="photo-tab" id="batch-tab-camera" onclick="switchBatchTab('camera')">📷 Camera</button>
          </div>
          <div id="batch-upload-pane">
            <div class="drop-zone" id="b-drop-zone" onclick="document.getElementById('b-photo').click()"
                 ondragover="event.preventDefault();this.classList.add('drag')"
                 ondragleave="this.classList.remove('drag')"
                 ondrop="bHandleDrop(event)">
              <input type="file" id="b-photo" accept="image/jpeg,image/png,image/webp" onchange="bHandleFileSelect(event)">
              <div class="drop-zone-icon" id="b-dz-icon">📷</div>
              <div class="drop-zone-label" id="b-dz-label">Drop group photo here or <span>browse</span></div>
            </div>
          </div>
          <div id="batch-camera-pane" style="display:none">
            <div class="cam-wrap">
              <video id="batch-cam-video" autoplay playsinline muted style="width:100%;height:100%;object-fit:cover;display:none"></video>
              <canvas id="batch-cam-canvas" style="display:none"></canvas>
              <div class="cam-overlay" id="batch-cam-overlay"><div class="cam-ring"></div><span>Camera off</span></div>
            </div>
            <div style="display:flex;gap:8px;margin-top:10px">
              <button class="btn btn-ghost btn-full" id="batch-cam-start-btn" onclick="batchStartCamera()">Start Camera</button>
              <button class="btn btn-primary btn-full" id="batch-cam-snap-btn" onclick="batchSnapPhoto()" disabled>Capture</button>
            </div>
            <img id="batch-cam-preview" class="cam-snap-preview" alt="Captured">
          </div>
          <div id="st-batch" class="status" style="margin-top:10px"></div>
          <div class="prog"><div class="prog-fill" id="bprog"></div></div>
          <button class="btn btn-green btn-full" id="btn-batch" onclick="submitBatch()" disabled>Process Group Photo</button>
        </div>
        <div id="batch-results" style="display:none" class="card">
          <div class="card-label">Results</div>
          <div class="stats" id="batch-stats"></div>
          <div class="blist" id="blist"></div>
        </div>
      </div>

      <!-- SESSIONS PANEL -->
      <div class="panel" id="panel-sessions">
        <div class="card">
          <div class="card-label">Create Session</div>
          <div class="split">
            <div>
              <div class="field"><label>University</label><select id="s-uni" onchange="sLoadDepts()"><option value="">— Select University —</option></select></div>
              <div class="field"><label>Department</label><select id="s-dept" onchange="sLoadUnits()" disabled><option value="">— Select Department —</option></select></div>
            </div>
            <div>
              <div class="field"><label>Unit</label><select id="s-unit" disabled><option value="">— Select Unit —</option></select></div>
              <div class="field"><label>Date *</label><input type="date" id="s-date"></div>
            </div>
          </div>
          <div class="split">
            <div class="field"><label>Start Time</label><input type="time" id="s-time"></div>
            <div class="field"><label>Room / Venue</label><input id="s-room" placeholder="e.g. LH-01"></div>
          </div>
          <div id="st-session" class="status"></div>
          <button class="btn btn-primary" onclick="createSession()">+ Create Session</button>
        </div>
        <div class="card">
          <div class="card-label">Recent Sessions</div>
          <div class="field"><label>Filter by Unit</label><select id="sf-unit" onchange="loadSessions()"><option value="">— Select Unit —</option></select></div>
          <div id="sessions-list"><div style="font-family:var(--font-m);font-size:.75rem;color:var(--muted)">Select a unit above to view sessions.</div></div>
        </div>
      </div>

      <!-- STUDENTS PANEL -->
      <div class="panel" id="panel-students">
        <div class="card">
          <div class="card-label">Enrolled Students</div>
          <div id="st-students" class="status show info">Loading students…</div>
          <div id="students-table-wrap" style="display:none">
            <div class="tbl-wrap">
              <table>
                <thead><tr><th>Student ID</th><th>Full Name</th><th>Program</th><th>Year</th><th>Face</th><th>Enrolled</th></tr></thead>
                <tbody id="students-tbody"></tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <!-- ANALYTICS PANEL -->
      <div class="panel" id="panel-analytics">
        <div class="card">
          <div class="card-label">Defaulters Report</div>
          <div class="split">
            <div class="field"><label>Year of Study</label>
              <select id="an-year">
                <option value="1">Year 1</option><option value="2">Year 2</option>
                <option value="3">Year 3</option><option value="4">Year 4</option>
              </select>
            </div>
            <div style="display:flex;align-items:flex-end;padding-bottom:11px">
              <button class="btn btn-primary btn-full" onclick="loadDefaulters()">Load Report</button>
            </div>
          </div>
          <div id="st-analytics" class="status"></div>
          <div id="defaulters-wrap" style="display:none">
            <div class="tbl-wrap">
              <table>
                <thead><tr><th>Name</th><th>Unit</th><th>Attendance %</th><th>Status</th><th>Exam Eligible</th></tr></thead>
                <tbody id="defaulters-tbody"></tbody>
              </table>
            </div>
          </div>
        </div>
        <div class="card">
          <div class="card-label">Department Census</div>
          <button class="btn btn-ghost" onclick="loadCensus()" style="margin-bottom:12px">Load Census</button>
          <div id="census-stats" class="stats" style="display:none"></div>
          <div id="st-census" class="status"></div>
        </div>
      </div>

      <!-- ADMIN LOGIN PANEL -->
      <div class="panel" id="panel-admin">
        <div class="card" style="max-width:420px">
          <div class="card-label">Admin Authentication</div>
          <div style="font-family:var(--font-d);font-size:1.1rem;font-weight:600;margin-bottom:4px">Sign in to UniAttend</div>
          <div style="font-family:var(--font-m);font-size:.72rem;color:var(--muted);margin-bottom:16px">Admin and HOD accounts only</div>
          <div class="field"><label>Email</label><input id="a-email" type="email" placeholder="admin@university.ac.ke" onkeydown="if(event.key==='Enter')doLogin()"></div>
          <div class="field"><label>Password</label><input id="a-pass" type="password" placeholder="••••••••" onkeydown="if(event.key==='Enter')doLogin()"></div>
          <div id="st-login" class="status"></div>
          <button class="btn btn-primary btn-full" id="btn-login" onclick="doLogin()">Sign In</button>
        </div>
      </div>

    </div>
  </div>
</div>
<script>
const BASE = window.location.origin;
let authToken = localStorage.getItem('ua_token') || null;
let authUser  = JSON.parse(localStorage.getItem('ua_user') || 'null');
let enrolFaceFile = null, enrolFaceBlob = null, enrolCamStream = null;
let batchPhotoFile = null, batchPhotoBlob = null, batchCamStream = null;

const panelTitles = {enrol:'Student Enrolment',batch:'Batch Attendance',sessions:'Sessions',students:'Students',analytics:'Analytics',admin:'Admin Login'};

window.addEventListener('DOMContentLoaded', () => { refreshAuthUI(); loadUniversities(); });

function showPanel(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const panel = document.getElementById('panel-' + name);
  const nav   = document.getElementById('nav-' + name);
  if (panel) panel.classList.add('active');
  if (nav)   nav.classList.add('active');
  document.getElementById('page-title').textContent = panelTitles[name] || '';
  closeSidebar();
  if (name === 'students' && authToken) loadStudents();
}

function openSidebar()  { document.getElementById('sidebar').classList.add('open'); document.getElementById('mob-overlay').classList.add('show'); }
function closeSidebar() { document.getElementById('sidebar').classList.remove('open'); document.getElementById('mob-overlay').classList.remove('show'); }

function refreshAuthUI() {
  const loggedIn = !!authToken && !!authUser;
  document.querySelectorAll('.admin-only').forEach(el => el.classList.toggle('visible', loggedIn));
  document.getElementById('sidebar-user').style.display  = loggedIn ? 'block' : 'none';
  document.getElementById('sidebar-guest').style.display = loggedIn ? 'none'  : 'block';
  if (loggedIn) {
    document.getElementById('user-display-name').textContent = authUser.username || 'Admin';
    document.getElementById('user-display-role').textContent = authUser.role || 'admin';
    document.getElementById('user-avatar-initial').textContent = (authUser.username || 'A')[0].toUpperCase();
    document.getElementById('nav-admin-label').textContent = 'Account';
  } else {
    document.getElementById('nav-admin-label').textContent = 'Admin Login';
  }
}

async function doLogin() {
  const email = document.getElementById('a-email').value.trim();
  const pass  = document.getElementById('a-pass').value;
  if (!email || !pass) return setStatus('st-login','err','Fill in email and password.');
  setStatus('st-login','info','Signing in…');
  document.getElementById('btn-login').disabled = true;
  try {
    const res = await fetch(`${BASE}/admin/login`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ email, password: pass })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Login failed');
    authToken = data.access_token;
    const vRes = await fetch(`${BASE}/admin/verify-token`, { headers: authHeaders() });
    const vData = await vRes.json();
    if (!vRes.ok) throw new Error('Token verification failed');
    authUser = { username: vData.username, role: vData.role, email };
    localStorage.setItem('ua_token', authToken);
    localStorage.setItem('ua_user', JSON.stringify(authUser));
    refreshAuthUI();
    setStatus('st-login','ok',`Welcome back, ${authUser.username}!`);
    setTimeout(() => showPanel('batch'), 900);
  } catch(e) {
    setStatus('st-login','err', e.message);
  } finally {
    document.getElementById('btn-login').disabled = false;
  }
}

function doLogout() {
  authToken = null; authUser = null;
  localStorage.removeItem('ua_token'); localStorage.removeItem('ua_user');
  refreshAuthUI(); showPanel('enrol');
}

function authHeaders() { return { 'Authorization': `Bearer ${authToken}`, 'Content-Type': 'application/json' }; }

async function loadUniversities() {
  try {
    const data = await apiFetch('/universities');
    ['e-uni','b-uni','s-uni'].forEach(id => populateSelect(id, data, 'id', 'name', '— Select University —'));
  } catch(e) { console.warn('Failed to load universities', e); }
}

async function eLoadDepts() {
  const uid = document.getElementById('e-uni').value;
  resetSelect('e-dept','— Select Department —'); resetSelect('e-unit','— Select Unit —');
  document.getElementById('e-dept').disabled = !uid; document.getElementById('e-unit').disabled = true;
  if (!uid) return;
  const data = await apiFetch(`/departments/${uid}`);
  populateSelect('e-dept', data, 'id', 'name', '— Select Department —');
  document.getElementById('e-dept').disabled = false;
}

// FIX: corrected eLoadUnits (was copy-pasted as duplicate of eLoadDepts with wrong vars)
async function eLoadUnits() {
  const did = document.getElementById('e-dept').value;
  resetSelect('e-unit','— Select Unit —');
  document.getElementById('e-unit').disabled = !did;
  if (!did) return;
  const data = await apiFetch(`/units/${did}`);
  populateSelect('e-unit', data, 'id', 'name', '— Select Unit —');
  document.getElementById('e-unit').disabled = false;
}

async function bLoadDepts() {
  const uid = document.getElementById('b-uni').value;
  resetSelect('b-dept','— Select Department —'); resetSelect('b-unit','— Select —'); resetSelect('b-session','— Select Session —');
  ['b-dept','b-unit','b-session'].forEach(id => document.getElementById(id).disabled = true);
  if (!uid) return;
  const data = await apiFetch(`/departments/${uid}`);
  populateSelect('b-dept', data, 'id', 'name', '— Select Department —');
  document.getElementById('b-dept').disabled = false;
}

async function bLoadUnits() {
  const did = document.getElementById('b-dept').value;
  resetSelect('b-unit','— Select —'); resetSelect('b-session','— Select Session —');
  ['b-unit','b-session'].forEach(id => document.getElementById(id).disabled = true);
  if (!did) return;
  const data = await apiFetch(`/units/${did}`);
  populateSelect('b-unit', data, 'id', 'name', '— Select —');
  document.getElementById('b-unit').disabled = false;
}

async function bLoadSessions() {
  const unitId = document.getElementById('b-unit').value;
  resetSelect('b-session','— Select Session —');
  document.getElementById('b-session').disabled = !unitId;
  if (!unitId) return;
  const data = await apiFetch(`/sessions/${unitId}`);
  const opts = data.map(s => ({ id: s.id, label: `${s.date}${s.start_time?' · '+s.start_time:''} ${s.room?'· '+s.room:''}` }));
  populateSelect('b-session', opts, 'id', 'label', '— Select Session —');
  document.getElementById('b-session').disabled = false;
  checkBatchReady();
}

async function sLoadDepts() {
  const uid = document.getElementById('s-uni').value;
  resetSelect('s-dept','— Select —'); resetSelect('s-unit','— Select —');
  ['s-dept','s-unit'].forEach(id => document.getElementById(id).disabled = true);
  if (!uid) return;
  const data = await apiFetch(`/departments/${uid}`);
  populateSelect('s-dept', data, 'id', 'name', '— Select —');
  document.getElementById('s-dept').disabled = false;
}

async function sLoadUnits() {
  const did = document.getElementById('s-dept').value;
  resetSelect('s-unit','— Select —');
  document.getElementById('s-unit').disabled = !did;
  if (!did) return;
  const data = await apiFetch(`/units/${did}`);
  populateSelect('s-unit', data, 'id', 'name', '— Select —');
  document.getElementById('s-unit').disabled = false;
  populateSelect('sf-unit', data, 'id', 'name', '— Select Unit —');
}

function enrolStep1() {
  const sid = document.getElementById('e-sid').value.trim();
  const name = document.getElementById('e-name').value.trim();
  if (!sid || !name) return setStatus('st-reg','err','Student ID and Full Name are required.');
  clearStatus('st-reg'); goEnrolStep(1);
}

function enrolStep2() {
  const uni = document.getElementById('e-uni').value;
  const dept = document.getElementById('e-dept').value;
  const unit = document.getElementById('e-unit').value;
  if (!uni || !dept || !unit) return setStatus('st-unit','err','Please select University, Department and Unit.');
  clearStatus('st-unit'); goEnrolStep(2);
}

function checkConsentReady() {
  const checked = document.getElementById('consent-check').checked;
  const sig     = document.getElementById('consent-signature').value.trim();
  const name    = document.getElementById('e-name').value.trim();
  const ready   = checked && sig.length >= 3;
  document.getElementById('btn-proceed-face').disabled = !ready;
  if (sig && name && sig.toLowerCase() !== name.toLowerCase()) {
    setStatus('st-consent','warn','Signature should match your full name above.');
  } else { clearStatus('st-consent'); }
}

function enrolStep3() {
  if (!document.getElementById('consent-signature').value.trim()) return;
  goEnrolStep(3);
}

function goEnrolStep(n) {
  for (let i = 0; i <= 4; i++) {
    const step = document.getElementById(`enrol-step${i}`);
    if (step) step.style.display = i === n ? '' : 'none';
    const dot = document.getElementById(`st${i}`);
    if (dot) { dot.classList.remove('active','done'); if (i < n) dot.classList.add('done'); else if (i === n) dot.classList.add('active'); }
  }
  if (n !== 3) stopEnrCamera();
}

function enrDrop(e) { e.preventDefault(); document.getElementById('enr-drop').classList.remove('drag'); const f = e.dataTransfer.files[0]; if(f) setEnrFile(f); }
function enrFileSelect(e) { if(e.target.files[0]) setEnrFile(e.target.files[0]); }
function setEnrFile(file) {
  enrolFaceFile = file; enrolFaceBlob = null;
  const prev = document.getElementById('e-preview');
  prev.src = URL.createObjectURL(file); prev.classList.add('show');
  setStatus('e-face-status','ok',`Photo selected: ${file.name}`);
  document.getElementById('btn-submit-enrol').disabled = false;
}

function switchEnrTab(tab) {
  document.getElementById('enr-upload-pane').style.display = tab==='upload' ? '' : 'none';
  document.getElementById('enr-camera-pane').style.display = tab==='camera' ? '' : 'none';
  document.getElementById('enr-tab-upload').classList.toggle('active', tab==='upload');
  document.getElementById('enr-tab-camera').classList.toggle('active', tab==='camera');
  if (tab !== 'camera') stopEnrCamera();
}

async function enrStartCamera() {
  try {
    enrolCamStream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'}});
    const vid = document.getElementById('enr-cam-video');
    vid.srcObject = enrolCamStream; vid.style.display = 'block';
    document.getElementById('enr-cam-overlay').style.display = 'none';
    document.getElementById('enr-cam-snap-btn').disabled = false;
    document.getElementById('enr-cam-start-btn').disabled = true;
  } catch(e) { setStatus('e-face-status','err','Camera access denied.'); }
}

function enrSnapPhoto() {
  const vid = document.getElementById('enr-cam-video');
  const cnv = document.getElementById('enr-cam-canvas');
  cnv.width = vid.videoWidth; cnv.height = vid.videoHeight;
  cnv.getContext('2d').drawImage(vid, 0, 0);
  cnv.toBlob(blob => {
    enrolFaceBlob = blob; enrolFaceFile = null;
    const url = URL.createObjectURL(blob);
    document.getElementById('e-preview').src = url;
    document.getElementById('e-preview').classList.add('show');
    document.getElementById('enr-cam-preview').src = url;
    document.getElementById('enr-cam-preview').style.display = 'block';
    setStatus('e-face-status','ok','Photo captured.');
    document.getElementById('btn-submit-enrol').disabled = false;
  }, 'image/jpeg', 0.92);
}

function stopEnrCamera() {
  if (enrolCamStream) { enrolCamStream.getTracks().forEach(t=>t.stop()); enrolCamStream = null; }
  const vid = document.getElementById('enr-cam-video');
  if (vid) { vid.style.display = 'none'; vid.srcObject = null; }
  const overlay = document.getElementById('enr-cam-overlay');
  if (overlay) overlay.style.display = 'flex';
  const snap = document.getElementById('enr-cam-snap-btn'); if (snap) snap.disabled = true;
  const start = document.getElementById('enr-cam-start-btn'); if (start) start.disabled = false;
}

async function submitEnrolment() {
  const sid    = document.getElementById('e-sid').value.trim();
  const name   = document.getElementById('e-name').value.trim();
  const email  = document.getElementById('e-email').value.trim();
  const phone  = document.getElementById('e-phone').value.trim();
  const prog   = document.getElementById('e-prog').value.trim();
  const year   = document.getElementById('e-year').value;
  const uniId  = document.getElementById('e-uni').value;
  const deptId = document.getElementById('e-dept').value;
  const unitId = document.getElementById('e-unit').value;
  if (!enrolFaceFile && !enrolFaceBlob) return setStatus('st-enrol','err','No face photo selected.');
  setStatus('st-enrol','info','Submitting enrolment…');
  document.getElementById('btn-submit-enrol').disabled = true;
  const fd = new FormData();
  fd.append('student_id', sid); fd.append('full_name', name); fd.append('email', email);
  fd.append('phone', phone); fd.append('program', prog); fd.append('year', year);
  fd.append('university_id', uniId); fd.append('department_id', deptId); fd.append('unit_id', unitId);
  fd.append('face_image', enrolFaceFile || new File([enrolFaceBlob], 'face.jpg', {type:'image/jpeg'}));
  try {
    const res  = await fetch(`${BASE}/enroll`, {method:'POST', body:fd});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Enrolment failed');
    document.getElementById('enrol-done-name').textContent = `${name} enrolled!`;
    document.getElementById('enrol-done-meta').textContent = `${sid} · ${prog} · Year ${year}`;
    goEnrolStep(4); stopEnrCamera();
  } catch(e) {
    setStatus('st-enrol','err', e.message);
    document.getElementById('btn-submit-enrol').disabled = false;
  }
}

function resetEnrol() {
  ['e-sid','e-name','e-email','e-phone','e-prog'].forEach(id => document.getElementById(id).value = '');
  document.getElementById('e-year').value = '1';
  resetSelect('e-uni','— Select University —'); resetSelect('e-dept','— Select Department —'); resetSelect('e-unit','— Select Unit —');
  document.getElementById('e-dept').disabled = true; document.getElementById('e-unit').disabled = true;
  document.getElementById('e-preview').classList.remove('show');
  document.getElementById('consent-check').checked = false;
  document.getElementById('consent-signature').value = '';
  document.getElementById('btn-proceed-face').disabled = true;
  enrolFaceFile = null; enrolFaceBlob = null; stopEnrCamera(); goEnrolStep(0);
}

function checkBatchReady() {
  const session = document.getElementById('b-session').value;
  const hasPhoto = !!(batchPhotoFile || batchPhotoBlob);
  document.getElementById('btn-batch').disabled = !(session && hasPhoto);
}

function switchBatchTab(tab) {
  document.getElementById('batch-upload-pane').style.display  = tab==='upload' ? '' : 'none';
  document.getElementById('batch-camera-pane').style.display  = tab==='camera' ? '' : 'none';
  document.getElementById('batch-tab-upload').classList.toggle('active', tab==='upload');
  document.getElementById('batch-tab-camera').classList.toggle('active', tab==='camera');
  if (tab !== 'camera') stopBatchCamera();
}

function bHandleDrop(e) { e.preventDefault(); document.getElementById('b-drop-zone').classList.remove('drag'); const f = e.dataTransfer.files[0]; if(f) setBatchFile(f); }
function bHandleFileSelect(e) { if(e.target.files[0]) setBatchFile(e.target.files[0]); }
function setBatchFile(file) {
  batchPhotoFile = file; batchPhotoBlob = null;
  document.getElementById('b-dz-icon').textContent = '✅';
  document.getElementById('b-dz-label').innerHTML = `<strong>${file.name}</strong>`;
  checkBatchReady();
}

async function batchStartCamera() {
  try {
    batchCamStream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}});
    const vid = document.getElementById('batch-cam-video');
    vid.srcObject = batchCamStream; vid.style.display = 'block';
    document.getElementById('batch-cam-overlay').style.display = 'none';
    document.getElementById('batch-cam-snap-btn').disabled = false;
    document.getElementById('batch-cam-start-btn').disabled = true;
  } catch(e) { setStatus('st-batch','err','Camera access denied.'); }
}

function batchSnapPhoto() {
  const vid = document.getElementById('batch-cam-video');
  const cnv = document.getElementById('batch-cam-canvas');
  cnv.width = vid.videoWidth; cnv.height = vid.videoHeight;
  cnv.getContext('2d').drawImage(vid, 0, 0);
  cnv.toBlob(blob => {
    batchPhotoBlob = blob; batchPhotoFile = null;
    document.getElementById('batch-cam-preview').src = URL.createObjectURL(blob);
    document.getElementById('batch-cam-preview').style.display = 'block';
    setStatus('st-batch','ok','Group photo captured. Ready to process.');
    checkBatchReady();
  }, 'image/jpeg', 0.92);
}

function stopBatchCamera() {
  if (batchCamStream) { batchCamStream.getTracks().forEach(t=>t.stop()); batchCamStream = null; }
  const vid = document.getElementById('batch-cam-video');
  if (vid) { vid.style.display = 'none'; vid.srcObject = null; }
  const overlay = document.getElementById('batch-cam-overlay');
  if (overlay) overlay.style.display = 'flex';
}

async function submitBatch() {
  if (!authToken) return setStatus('st-batch','err','Please log in as admin first.');
  const sessionId = document.getElementById('b-session').value;
  if (!sessionId) return setStatus('st-batch','err','Select a session first.');
  if (!batchPhotoFile && !batchPhotoBlob) return setStatus('st-batch','err','No photo selected.');
  setStatus('st-batch','info','Processing group photo…');
  document.getElementById('btn-batch').disabled = true;
  setProg('bprog', 30);
  const fd = new FormData();
  fd.append('session_id', sessionId);
  fd.append('group_photo', batchPhotoFile || new File([batchPhotoBlob], 'group.jpg', {type:'image/jpeg'}));
  try {
    const res = await fetch(`${BASE}/attendance/batch`, {
      method:'POST', headers:{'Authorization': `Bearer ${authToken}`}, body: fd
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Batch failed');
    setProg('bprog', 100);
    setStatus('st-batch','ok',`Done — ${data.students_matched} matched, ${data.unmatched_faces} unrecognised.`);
    renderBatchResults(data);
  } catch(e) {
    setStatus('st-batch','err', e.message); setProg('bprog', 0);
  } finally {
    document.getElementById('btn-batch').disabled = false;
  }
}

function renderBatchResults(data) {
  document.getElementById('batch-results').style.display = '';
  document.getElementById('batch-stats').innerHTML = `
    <div class="stat"><div class="stat-val">${data.faces_detected}</div><div class="stat-lbl">Detected</div></div>
    <div class="stat"><div class="stat-val" style="color:var(--green)">${data.students_matched}</div><div class="stat-lbl">Matched</div></div>
    <div class="stat"><div class="stat-val" style="color:var(--amber)">${data.unmatched_faces}</div><div class="stat-lbl">Unmatched</div></div>
  `;
  document.getElementById('blist').innerHTML = (data.faces || []).map(f => `
    <div class="bi">
      <span>${f.student_id || '—'}</span>
      <span style="color:var(--muted);font-size:.68rem">score: ${f.score}</span>
      <span class="pill ${f.status==='verified'?'pill-ok':f.status==='duplicate'?'pill-warn':'pill-fail'}">${f.status}</span>
    </div>
  `).join('') || '<div class="bi" style="color:var(--muted)">No face data.</div>';
}

async function createSession() {
  if (!authToken) return setStatus('st-session','err','Admin login required.');
  const unitId = document.getElementById('s-unit').value;
  const date   = document.getElementById('s-date').value;
  if (!unitId || !date) return setStatus('st-session','err','Unit and date are required.');
  setStatus('st-session','info','Creating session…');
  try {
    const res = await fetch(`${BASE}/sessions`, {
      method:'POST', headers: authHeaders(),
      body: JSON.stringify({ unit_id: unitId, date, start_time: document.getElementById('s-time').value || null, room: document.getElementById('s-room').value.trim() || null })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to create session');
    setStatus('st-session','ok','Session created.');
    loadSessions();
  } catch(e) { setStatus('st-session','err', e.message); }
}

async function loadSessions() {
  const unitId = document.getElementById('sf-unit').value;
  if (!unitId) return;
  const data = await apiFetch(`/sessions/${unitId}`);
  const container = document.getElementById('sessions-list');
  if (!data.length) { container.innerHTML = '<div style="font-family:var(--font-m);font-size:.75rem;color:var(--muted)">No sessions found for this unit.</div>'; return; }
  container.innerHTML = `<div style="border:1.5px solid var(--border);border-radius:var(--r);overflow:hidden">` +
    data.map(s => `
      <div class="sess-row">
        <div>
          <div style="font-family:var(--font-m);font-size:.78rem">${s.date}${s.start_time ? ' · ' + s.start_time : ''}</div>
          <div class="sess-meta">${s.room || 'No room set'} · ID: ${s.id.slice(0,8)}…</div>
        </div>
        <button class="btn btn-danger" style="padding:5px 12px;font-size:.72rem" onclick="deleteSession('${s.id}')">Delete</button>
      </div>
    `).join('') + `</div>`;
}

async function deleteSession(id) {
  if (!confirm('Delete this session? This cannot be undone.')) return;
  try {
    const res = await fetch(`${BASE}/sessions/${id}`, { method:'DELETE', headers: authHeaders() });
    if (!res.ok) throw new Error('Failed to delete');
    loadSessions();
  } catch(e) { alert(e.message); }
}

async function loadStudents() {
  setStatus('st-students','info','Loading students…');
  try {
    const data = await apiFetch('/students', authHeaders());
    clearStatus('st-students');
    document.getElementById('students-table-wrap').style.display = '';
    document.getElementById('students-tbody').innerHTML = data.map(s => `
      <tr>
        <td>${s.student_id || '—'}</td><td>${s.full_name || '—'}</td>
        <td>${s.program || '—'}</td><td>${s.year || '—'}</td>
        <td><span class="badge ${s.has_face?'b-yes':'b-no'}">${s.has_face?'✓ Yes':'✗ No'}</span></td>
        <td>${s.enrolled_at ? new Date(s.enrolled_at).toLocaleDateString('en-KE') : '—'}</td>
      </tr>
    `).join('') || '<tr><td colspan="6" style="text-align:center;color:var(--muted)">No students found.</td></tr>';
  } catch(e) { setStatus('st-students','err','Failed to load students.'); }
}

// FIX: removed duplicate try block; defaulters-wrap now correctly shown on success
async function loadDefaulters() {
  const year = document.getElementById('an-year').value;
  setStatus('st-analytics','info','Loading report…');
  document.getElementById('defaulters-wrap').style.display = 'none';
  try {
    const data = await apiFetch(`/analytics/defaulters/${year}`, authHeaders());
    clearStatus('st-analytics');
    if (!data.length) { setStatus('st-analytics','ok','No defaulters found for this year.'); return; }
    document.getElementById('defaulters-wrap').style.display = '';
    document.getElementById('defaulters-tbody').innerHTML = data.map(d => `
      <tr>
        <td>${d.name}</td><td style="font-size:.7rem">${d.unit}</td><td>${d.percentage}</td>
        <td><span class="badge ${d.status==='DANGER'?'b-fail':'b-no'}">${d.status}</span></td>
        <td><span class="badge ${d.exam_eligible?'b-yes':'b-fail'}">${d.exam_eligible?'Yes':'No'}</span></td>
      </tr>
    `).join('');
  } catch(e) { setStatus('st-analytics','err','Failed to load report.'); }
}

async function loadCensus() {
  setStatus('st-census','info','Loading census…');
  try {
    const data = await apiFetch('/analytics/department-census', authHeaders());
    clearStatus('st-census');
    const s = data.summary;
    document.getElementById('census-stats').style.display = 'grid';
    document.getElementById('census-stats').innerHTML = `
      <div class="stat"><div class="stat-val">${s.total_students}</div><div class="stat-lbl">Total Students</div></div>
      <div class="stat"><div class="stat-val" style="color:var(--green)">${s.active_today}</div><div class="stat-lbl">Active Today</div></div>
      <div class="stat"><div class="stat-val" style="color:var(--muted)">${s.period}</div><div class="stat-lbl">Period</div></div>
    `;
  } catch(e) { setStatus('st-census','err','Failed to load census.'); }
}

async function apiFetch(path, headers={}) {
  const res = await fetch(`${BASE}${path}`, { headers });
  if (!res.ok) { const e = await res.json().catch(()=>{}); throw new Error(e?.detail || `HTTP ${res.status}`); }
  return res.json();
}
function setStatus(id, type, msg) { const el = document.getElementById(id); if(!el) return; el.className=`status show ${type}`; el.innerHTML=`<span>${msg}</span>`; }
function clearStatus(id) { const el = document.getElementById(id); if(el) el.className='status'; }
function setProg(id, pct) { const el = document.getElementById(id); if(el) el.style.width=pct+'%'; }
function populateSelect(id, items, valKey, labelKey, placeholder) {
  const sel = document.getElementById(id); if(!sel) return;
  sel.innerHTML = `<option value="">${placeholder}</option>` + items.map(i => `<option value="${i[valKey]}">${i[labelKey]}</option>`).join('');
}
function resetSelect(id, placeholder) { const sel = document.getElementById(id); if(sel) { sel.innerHTML=`<option value="">${placeholder}</option>`; sel.disabled=true; } }
</script>
</body>
</html>"""
