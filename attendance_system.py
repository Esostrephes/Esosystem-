#!/usr/bin/env python3
# ==============================
# UNIVERSITY ATTENDANCE SYSTEM
# FastAPI + Supabase + Redis
# REAL Liveness (MediaPipe Head Movement in Browser)
# JWT Admin Auth
# BATCH ATTENDANCE (Sequential for 1-100 students)
# ==============================
#
# SUPABASE SETUP — run these in Supabase SQL Editor before starting:
#
#   CREATE TABLE students (
#       id          BIGSERIAL PRIMARY KEY,
#       student_id  TEXT UNIQUE NOT NULL,
#       created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
#   );
#
#   CREATE TABLE attendance (
#       id          BIGSERIAL PRIMARY KEY,
#       student_id  TEXT NOT NULL,
#       timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
#       marked_by   TEXT DEFAULT 'self'
#   );
#   CREATE INDEX idx_att_student   ON attendance(student_id);
#   CREATE INDEX idx_att_timestamp ON attendance(timestamp);
#
#   CREATE TABLE admins (
#       id            BIGSERIAL PRIMARY KEY,
#       username      TEXT UNIQUE NOT NULL,
#       password_hash TEXT NOT NULL
#   );
#
# Required env vars:
#   SUPABASE_URL      — e.g. https://xxxx.supabase.co
#   SUPABASE_KEY      — service_role secret key  (NOT the anon key)
#   REDIS_HOST        — optional, falls back to in-memory
#   JWT_SECRET        — your own secret string
# ==============================

from fastapi import FastAPI, Form, HTTPException, Request, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client
import redis
import random
import uuid
import os
import time
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- CONFIG ----------------

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")   # use service_role key

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY environment variables are required")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
JWT_SECRET = os.getenv("JWT_SECRET", "super_secret_jwt_key_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

# ---------------- INIT ----------------

app = FastAPI(title="University Attendance Pilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Supabase client (singleton) ----
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
logger.info("Supabase client initialised.")

# ---- Redis connection with graceful degradation ----
_redis_client: Optional[redis.Redis] = None

def get_redis() -> Optional[redis.Redis]:
    """Return Redis client, or None if unavailable."""
    global _redis_client
    if _redis_client is None:
        try:
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            client.ping()
            _redis_client = client
            logger.info("Redis connected.")
        except redis.RedisError as e:
            logger.warning(f"Redis unavailable: {e}. Challenges will use in-memory fallback.")
            return None
    return _redis_client

# In-memory fallback for challenges (single-process only)
_challenge_store: Dict[str, dict] = {}

def challenge_set(key: str, value: str, ex: int) -> None:
    r = get_redis()
    if r:
        r.set(key, value, ex=ex)
    else:
        _challenge_store[key] = {"value": value, "expires": time.time() + ex}

def challenge_get(key: str) -> Optional[str]:
    r = get_redis()
    if r:
        return r.get(key)
    entry = _challenge_store.get(key)
    if entry and time.time() < entry["expires"]:
        return entry["value"]
    _challenge_store.pop(key, None)
    return None

def challenge_delete(key: str) -> None:
    r = get_redis()
    if r:
        r.delete(key)
    else:
        _challenge_store.pop(key, None)

def cache_set(key: str, value: str, ex: int) -> None:
    r = get_redis()
    if r:
        r.set(key, value, ex=ex)

# Auth setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/login")

# Tables are created via Supabase SQL Editor (see top-of-file instructions).
# No runtime schema migration needed.

# ---------------- AUTH ----------------

def create_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)

def get_current_admin(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

# ---------------- ADMIN ROUTES ----------------

@app.post("/admin/create")
def create_admin(username: str = Form(...), password: str = Form(...)):
    if len(username) < 3 or len(password) < 6:
        raise HTTPException(400, "Username ≥3 chars, password ≥6 chars")
    hashed = pwd_context.hash(password)
    try:
        supabase.table("admins").insert({
            "username": username,
            "password_hash": hashed,
        }).execute()
    except Exception as e:
        if "duplicate" in str(e).lower() or "unique" in str(e).lower():
            raise HTTPException(400, "Admin already exists")
        raise HTTPException(500, f"Database error: {e}")
    return {"status": "admin_created"}

@app.post("/admin/login")
def admin_login(form_data: OAuth2PasswordRequestForm = Depends()):
    result = (
        supabase.table("admins")
        .select("password_hash")
        .eq("username", form_data.username)
        .maybe_single()
        .execute()
    )
    row = result.data
    if not row or not pwd_context.verify(form_data.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token({"sub": form_data.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/admin/verify-token")
def verify_token(current_admin: str = Depends(get_current_admin)):
    """Endpoint to validate a stored token without side effects."""
    return {"valid": True, "username": current_admin}

@app.get("/analytics", response_model=Dict)
def analytics(current_admin: str = Depends(get_current_admin)):
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    total_res = supabase.table("attendance").select("id", count="exact").execute()
    total = total_res.count or 0

    # Count unique students (fetch all ids, deduplicate in Python — Supabase free tier
    # doesn't expose COUNT(DISTINCT) directly via the PostgREST API)
    unique_res = supabase.table("attendance").select("student_id").execute()
    unique_students = len({row["student_id"] for row in (unique_res.data or [])})

    today_res = (
        supabase.table("attendance")
        .select("id", count="exact")
        .gte("timestamp", today_start)
        .execute()
    )
    today_count = today_res.count or 0

    return {
        "total_attendance_records": total,
        "unique_students_all_time": unique_students,
        "today_count": today_count,
        "admin": current_admin,
    }

@app.get("/students", response_model=List[Dict])
def get_students(current_admin: str = Depends(get_current_admin)):
    result = (
        supabase.table("students")
        .select("student_id, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []

# ---------------- REGISTRATION ----------------

@app.post("/register")
def register(student_id: str = Form(...)):
    student_id = student_id.strip()
    if not student_id or len(student_id) > 100:
        raise HTTPException(400, "Invalid student ID")
    try:
        supabase.table("students").insert({
            "student_id": student_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        if "duplicate" in str(e).lower() or "unique" in str(e).lower():
            raise HTTPException(400, "Student already registered")
        raise HTTPException(500, f"Database error: {e}")
    return {"status": "registered", "student_id": student_id}

@app.post("/challenge")
def generate_challenge(student_id: str = Form(...)):
    """Generate a unique liveness challenge per student session."""
    student_id = student_id.strip()
    # Verify student exists
    result = (
        supabase.table("students")
        .select("student_id")
        .eq("student_id", student_id)
        .maybe_single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Student not registered")

    challenges = ["LEFT", "RIGHT", "UP"]
    challenge = random.choice(challenges)
    challenge_id = str(uuid.uuid4())[:12]
    key = f"challenge:{student_id}:{challenge_id}"
    challenge_set(key, challenge, ex=120)  # 2-minute expiry
    return {"challenge": challenge, "challenge_id": challenge_id, "student_id": student_id}

# ---------------- MARK ATTENDANCE ----------------

@app.post("/mark_attendance")
def mark_attendance(
    student_id: str = Form(...),
    movement: str = Form(...),
    challenge_id: str = Form(...),
):
    student_id = student_id.strip()
    movement = movement.strip().upper()

    key = f"challenge:{student_id}:{challenge_id}"
    expected = challenge_get(key)
    challenge_delete(key)  # Consume regardless — prevents replay attacks

    if not expected:
        raise HTTPException(status_code=400, detail="Challenge expired or invalid")

    if movement != expected.upper():
        raise HTTPException(status_code=403, detail="Liveness check failed")

    # Verify student exists
    student_res = (
        supabase.table("students")
        .select("student_id")
        .eq("student_id", student_id)
        .maybe_single()
        .execute()
    )
    if not student_res.data:
        raise HTTPException(status_code=404, detail="Student not registered")

    supabase.table("attendance").insert({
        "student_id": student_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "marked_by": "self",
    }).execute()

    cache_set(f"last_seen:{student_id}", str(datetime.now()), ex=86400)
    return {"status": "attendance_marked", "student_id": student_id}

# ---------------- FRONTEND ----------------

@app.get("/", response_class=HTMLResponse)
def home():
    # NOTE: HTML is kept in a separate variable to avoid f-string/template conflicts
    return HTML_PAGE

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UniAttend — University Attendance</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.js" crossorigin="anonymous"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0c0e14;
    --surface: #131620;
    --surface2: #1a1e2e;
    --border: rgba(255,255,255,0.07);
    --accent: #5b6eff;
    --accent2: #00e5a0;
    --accent3: #ff5b6e;
    --text: #e8eaf0;
    --muted: #6b7280;
    --font-display: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
    --radius: 10px;
    --radius-lg: 16px;
  }

  body {
    font-family: var(--font-display);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(91,110,255,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(91,110,255,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .wrapper {
    position: relative;
    z-index: 1;
    max-width: 860px;
    margin: 0 auto;
    padding: 32px 20px 80px;
  }

  /* Header */
  header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 40px;
  }

  .logo-mark {
    width: 46px;
    height: 46px;
    background: var(--accent);
    border-radius: 12px;
    display: grid;
    place-items: center;
    font-size: 22px;
    flex-shrink: 0;
    box-shadow: 0 0 24px rgba(91,110,255,0.4);
  }

  header h1 {
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -0.03em;
  }

  header p {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 2px;
    letter-spacing: 0.05em;
  }

  .header-badge {
    margin-left: auto;
    background: rgba(0,229,160,0.12);
    border: 1px solid rgba(0,229,160,0.25);
    color: var(--accent2);
    font-family: var(--font-mono);
    font-size: 0.68rem;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
  }

  /* Tabs */
  .tabs {
    display: flex;
    gap: 2px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 4px;
    margin-bottom: 28px;
  }

  .tab-btn {
    flex: 1;
    padding: 10px 0;
    border: none;
    background: transparent;
    color: var(--muted);
    font-family: var(--font-display);
    font-size: 0.85rem;
    font-weight: 600;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.02em;
  }

  .tab-btn:hover { color: var(--text); background: rgba(255,255,255,0.04); }
  .tab-btn.active { background: var(--accent); color: #fff; box-shadow: 0 4px 16px rgba(91,110,255,0.35); }

  /* Panels */
  .panel { display: none; }
  .panel.active { display: block; animation: fadeIn 0.25s ease; }

  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; } }

  /* Card */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    margin-bottom: 16px;
  }

  .card-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 16px;
  }

  /* Camera */
  .camera-wrap {
    position: relative;
    border-radius: var(--radius);
    overflow: hidden;
    background: #000;
    aspect-ratio: 4/3;
    max-width: 380px;
  }

  .camera-wrap video {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transform: scaleX(-1); /* Mirror effect */
  }

  .camera-overlay {
    position: absolute;
    inset: 0;
    border: 2px solid rgba(91,110,255,0.3);
    border-radius: var(--radius);
    pointer-events: none;
  }

  .camera-corners::before, .camera-corners::after {
    content: '';
    position: absolute;
    width: 24px;
    height: 24px;
    border-color: var(--accent);
    border-style: solid;
  }

  .camera-corners::before {
    top: 8px; left: 8px;
    border-width: 2px 0 0 2px;
  }

  .camera-corners::after {
    bottom: 8px; right: 8px;
    border-width: 0 2px 2px 0;
  }

  /* Form controls */
  .field { margin-bottom: 12px; }
  .field label {
    display: block;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    margin-bottom: 6px;
    text-transform: uppercase;
  }

  input, textarea {
    width: 100%;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 0.9rem;
    padding: 11px 14px;
    outline: none;
    transition: border-color 0.2s;
  }

  input:focus, textarea:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(91,110,255,0.12);
  }

  textarea { resize: vertical; min-height: 120px; }

  /* Buttons */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 11px 22px;
    border: none;
    border-radius: var(--radius);
    font-family: var(--font-display);
    font-size: 0.88rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.02em;
  }

  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover:not(:disabled) { background: #4a5dff; box-shadow: 0 4px 20px rgba(91,110,255,0.4); transform: translateY(-1px); }
  .btn-success { background: var(--accent2); color: #0c0e14; }
  .btn-success:hover:not(:disabled) { box-shadow: 0 4px 20px rgba(0,229,160,0.35); transform: translateY(-1px); }
  .btn-ghost { background: transparent; border: 1px solid var(--border); color: var(--text); }
  .btn-ghost:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
  .btn-danger { background: transparent; border: 1px solid rgba(255,91,110,0.3); color: var(--accent3); }
  .btn-danger:hover { background: rgba(255,91,110,0.1); }
  .btn-full { width: 100%; justify-content: center; }

  /* Challenge display */
  .challenge-display {
    margin: 16px 0;
    padding: 20px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    text-align: center;
    min-height: 88px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 6px;
  }

  .challenge-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .challenge-action {
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--accent);
    letter-spacing: -0.02em;
    line-height: 1;
  }

  .challenge-hint {
    font-size: 0.72rem;
    color: var(--muted);
    font-family: var(--font-mono);
  }

  /* Status toast */
  .status-bar {
    padding: 12px 16px;
    border-radius: var(--radius);
    font-size: 0.85rem;
    display: none;
    margin: 12px 0;
    align-items: center;
    gap: 8px;
  }

  .status-bar.show { display: flex; animation: fadeIn 0.2s; }
  .status-bar.ok { background: rgba(0,229,160,0.1); border: 1px solid rgba(0,229,160,0.25); color: var(--accent2); }
  .status-bar.err { background: rgba(255,91,110,0.1); border: 1px solid rgba(255,91,110,0.25); color: var(--accent3); }
  .status-bar.info { background: rgba(91,110,255,0.1); border: 1px solid rgba(91,110,255,0.25); color: var(--accent); }

  /* Layout split */
  .split {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  @media (max-width: 620px) { .split { grid-template-columns: 1fr; } }

  /* Batch list */
  .batch-list {
    max-height: 280px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    scrollbar-width: thin;
    scrollbar-color: var(--accent) transparent;
  }

  .batch-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 0.82rem;
  }

  .batch-item:last-child { border-bottom: none; }
  .batch-item.active { background: rgba(91,110,255,0.08); }

  .batch-pill {
    font-size: 0.68rem;
    padding: 3px 9px;
    border-radius: 20px;
    font-weight: 500;
  }

  .pill-pending { background: rgba(255,255,255,0.06); color: var(--muted); }
  .pill-active { background: rgba(91,110,255,0.2); color: var(--accent); }
  .pill-ok { background: rgba(0,229,160,0.15); color: var(--accent2); }
  .pill-fail { background: rgba(255,91,110,0.15); color: var(--accent3); }

  /* Admin analytics grid */
  .analytics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 20px;
  }

  @media (max-width: 500px) { .analytics-grid { grid-template-columns: 1fr 1fr; } }

  .stat-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
  }

  .stat-card .value {
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--accent);
  }

  .stat-card .label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }

  .students-table { width: 100%; border-collapse: collapse; margin-top: 8px; }
  .students-table th {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }

  .students-table td {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
  }

  .students-table tr:last-child td { border-bottom: none; }
  .students-table tr:hover td { background: rgba(255,255,255,0.02); }

  .table-wrap {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    max-height: 300px;
    overflow-y: auto;
  }

  /* Progress bar for batch */
  .progress-bar {
    height: 3px;
    background: var(--surface2);
    border-radius: 2px;
    overflow: hidden;
    margin: 12px 0;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 2px;
    transition: width 0.4s ease;
    width: 0%;
  }

  /* Pulse animation for active detection */
  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(91,110,255,0.4); }
    50% { box-shadow: 0 0 0 10px rgba(91,110,255,0); }
  }

  .detecting .camera-overlay { animation: pulse 1.5s infinite; border-color: var(--accent); }
  .detected .camera-overlay { border-color: var(--accent2); }

  /* Divider */
  .divider { height: 1px; background: var(--border); margin: 20px 0; }

  /* Spinner */
  .spinner {
    width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,0.25);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    flex-shrink: 0;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  /* Loader overlay while MP initializes */
  #mp-loading {
    position: fixed;
    inset: 0;
    background: var(--bg);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    z-index: 999;
    transition: opacity 0.4s;
  }

  #mp-loading.hidden { opacity: 0; pointer-events: none; }
  #mp-loading .lbl { font-family: var(--font-mono); font-size: 0.8rem; color: var(--muted); }
  #mp-loading .bar { width: 200px; height: 3px; background: var(--surface2); border-radius: 2px; overflow: hidden; }
  #mp-loading .bar-fill { height: 100%; background: var(--accent); border-radius: 2px; animation: mpLoad 2s ease forwards; }
  @keyframes mpLoad { from { width: 0; } to { width: 90%; } }
</style>
</head>
<body>

<div id="mp-loading">
  <div style="font-size:2rem;">👁</div>
  <div class="lbl">Loading face detection model...</div>
  <div class="bar"><div class="bar-fill"></div></div>
</div>

<div class="wrapper">
  <header>
    <div class="logo-mark">🎓</div>
    <div>
      <h1>UniAttend</h1>
      <p>LIVENESS-VERIFIED ATTENDANCE SYSTEM • PILOT v2</p>
    </div>
    <div class="header-badge">● LIVE</div>
  </header>

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab(0)">Individual</button>
    <button class="tab-btn" onclick="switchTab(1)">Batch Mode</button>
    <button class="tab-btn" onclick="switchTab(2)">Admin</button>
  </div>

  <!-- ===== INDIVIDUAL ===== -->
  <div id="panel0" class="panel active">
    <div class="split">
      <div>
        <div class="card-title">Camera Feed</div>
        <div class="camera-wrap" id="cameraWrap0">
          <video id="video0" autoplay playsinline muted></video>
          <div class="camera-overlay camera-corners"></div>
        </div>
      </div>
      <div>
        <div class="card-title">Student Details</div>
        <div class="field">
          <label>Student ID</label>
          <input id="sid_individual" placeholder="e.g. U2023001" autocomplete="off">
        </div>

        <div class="challenge-display" id="challenge_box">
          <div class="challenge-label">Challenge</div>
          <div class="challenge-action" id="challenge_text">—</div>
          <div class="challenge-hint" id="challenge_hint">Press Start to begin</div>
        </div>

        <div id="status_individual" class="status-bar"></div>

        <button class="btn btn-primary btn-full" id="btn_start" onclick="startIndividual()">
          ▶ Start Attendance
        </button>
      </div>
    </div>

    <div class="divider"></div>
    <div class="card-title">Not registered?</div>
    <div style="display:flex; gap:8px;">
      <input id="reg_sid" placeholder="Student ID to register" style="flex:1;">
      <button class="btn btn-ghost" onclick="registerStudent()">Register</button>
    </div>
    <div id="status_register" class="status-bar" style="margin-top:10px;"></div>
  </div>

  <!-- ===== BATCH ===== -->
  <div id="panel1" class="panel">
    <div class="card">
      <div class="card-title">Student List</div>
      <div class="field">
        <label>Student IDs — one per line</label>
        <textarea id="batch_ids" placeholder="U2023001&#10;U2023002&#10;U2023003&#10;..."></textarea>
      </div>
      <button class="btn btn-success btn-full" id="btn_batch" onclick="startBatch()">
        🚀 Start Batch Attendance
      </button>
    </div>

    <div id="status_batch" class="status-bar"></div>
    <div class="progress-bar"><div class="progress-fill" id="batch_progress"></div></div>

    <div style="display:flex; gap:8px; margin-bottom: 16px; align-items:center;">
      <div class="camera-wrap" style="max-width:200px; flex-shrink:0;">
        <video id="video1" autoplay playsinline muted></video>
        <div class="camera-overlay camera-corners" id="cameraWrap1overlay"></div>
      </div>
      <div style="flex:1;">
        <div class="challenge-display" id="batch_challenge_box" style="margin:0;">
          <div class="challenge-label">Current Challenge</div>
          <div class="challenge-action" id="batch_challenge_text">—</div>
          <div class="challenge-hint" id="batch_challenge_hint">Waiting...</div>
        </div>
      </div>
    </div>

    <div class="batch-list" id="batch_list"></div>
  </div>

  <!-- ===== ADMIN ===== -->
  <div id="panel2" class="panel">
    <div id="login_section">
      <div class="card">
        <div class="card-title">Admin Login</div>
        <div class="field">
          <label>Username</label>
          <input id="admin_user" placeholder="admin" autocomplete="username">
        </div>
        <div class="field">
          <label>Password</label>
          <input id="admin_pass" type="password" placeholder="••••••••" autocomplete="current-password">
        </div>
        <button class="btn btn-primary btn-full" onclick="adminLogin()">Login</button>
        <div id="status_login" class="status-bar" style="margin-top:12px;"></div>
      </div>
    </div>

    <div id="admin_section" style="display:none;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
        <div>
          <div class="card-title" style="margin:0;">Dashboard</div>
          <div style="font-family:var(--font-mono); font-size:0.72rem; color:var(--muted);" id="admin_greeting"></div>
        </div>
        <div style="display:flex; gap:8px;">
          <button class="btn btn-ghost" onclick="refreshAdmin()">↻ Refresh</button>
          <button class="btn btn-danger" onclick="adminLogout()">Logout</button>
        </div>
      </div>

      <div class="analytics-grid" id="analytics_grid">
        <div class="stat-card"><div class="value" id="s_total">—</div><div class="label">Total Records</div></div>
        <div class="stat-card"><div class="value" id="s_today">—</div><div class="label">Today</div></div>
        <div class="stat-card"><div class="value" id="s_unique">—</div><div class="label">Unique Students</div></div>
      </div>

      <div class="card-title">Registered Students</div>
      <div class="table-wrap">
        <table class="students-table">
          <thead><tr><th>Student ID</th><th>Registered At</th></tr></thead>
          <tbody id="students_tbody"><tr><td colspan="2" style="color:var(--muted);padding:16px;">Loading...</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- ===== SHARED HIDDEN VIDEO (used as fallback) ===== -->
<video id="video_shared" autoplay playsinline muted style="display:none;"></video>

<script>
// ================================================================
// MEDIAPIPE + CORE SETUP
// ================================================================
let faceLandmarker = null;
let streams = {};          // { 0: MediaStream, 1: MediaStream }
let detecting = false;
let noseHistory = [];
const THRESHOLD = 22;
const HISTORY = 15;
const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MP_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

async function initMediaPipe() {
  // The CDN exposes `vision` as a global
  const { FaceLandmarker, FilesetResolver } = window.vision;
  const fs = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  faceLandmarker = await FaceLandmarker.createFromOptions(fs, {
    baseOptions: { modelAssetPath: MP_MODEL },
    runningMode: "VIDEO",
    numFaces: 1,
  });
  console.log("✅ MediaPipe ready");
}

async function startCamera(videoEl, cameraWrap) {
  const existingStream = streams[videoEl.id];
  if (existingStream) return; // Already running
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    });
    videoEl.srcObject = stream;
    streams[videoEl.id] = stream;
    await videoEl.play();
  } catch (e) {
    console.error("Camera error:", e);
    throw e;
  }
}

function stopCamera(videoEl) {
  const stream = streams[videoEl.id];
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    videoEl.srcObject = null;
    delete streams[videoEl.id];
  }
}

function stopDetection() {
  detecting = false;
}

function detectHeadMovement(videoEl, challenge, onSuccess, onTimeout) {
  if (!faceLandmarker || !videoEl) return;
  detecting = true;
  noseHistory = [];
  let startTime = performance.now();
  const TIMEOUT_MS = 30000; // 30s per student

  const loop = () => {
    if (!detecting) return;
    if (performance.now() - startTime > TIMEOUT_MS) {
      detecting = false;
      if (onTimeout) onTimeout();
      return;
    }

    try {
      const result = faceLandmarker.detectForVideo(videoEl, performance.now());
      if (result.faceLandmarks && result.faceLandmarks.length > 0) {
        const nose = result.faceLandmarks[0][1]; // index 1 = nose tip
        const w = videoEl.videoWidth || 640;
        const h = videoEl.videoHeight || 480;
        noseHistory.push({ x: nose.x * w, y: nose.y * h });
        if (noseHistory.length > HISTORY) noseHistory.shift();

        if (noseHistory.length >= 8) {
          const s = noseHistory[0];
          const e = noseHistory[noseHistory.length - 1];
          const dx = e.x - s.x;
          const dy = e.y - s.y;
          let detected = null;
          if (Math.abs(dx) > THRESHOLD) {
            // Mirror-compensated: positive dx means head went LEFT in mirror view
            detected = dx > 0 ? "LEFT" : "RIGHT";
          } else if (Math.abs(dy) > THRESHOLD * 0.8) {
            detected = dy < 0 ? "UP" : "DOWN";
          }
          if (detected && detected === challenge.toUpperCase()) {
            detecting = false;
            onSuccess();
            return;
          }
        }
      }
    } catch (e) {
      // Frame not ready yet
    }
    requestAnimationFrame(loop);
  };
  loop();
}

// ================================================================
// UTILS
// ================================================================
function showStatus(id, msg, type) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.className = "status-bar show " + (type === "ok" ? "ok" : type === "err" ? "err" : "info");
}

function hideStatus(id) {
  document.getElementById(id).className = "status-bar";
}

async function apiFetch(path, opts = {}) {
  const res = await fetch(path, opts);
  let data;
  try { data = await res.json(); } catch { data = {}; }
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

function formData(obj) {
  const fd = new FormData();
  Object.entries(obj).forEach(([k, v]) => fd.append(k, v));
  return fd;
}

// ================================================================
// TAB SWITCHING
// ================================================================
function switchTab(i) {
  document.querySelectorAll(".panel").forEach((p, j) => p.classList.toggle("active", i === j));
  document.querySelectorAll(".tab-btn").forEach((b, j) => b.classList.toggle("active", i === j));
  // Start camera for the active tab
  if (i === 0) ensureCamera("video0");
  if (i === 1) ensureCamera("video1");
}

async function ensureCamera(videoId) {
  const v = document.getElementById(videoId);
  if (!streams[videoId]) {
    try {
      await startCamera(v);
    } catch (e) {
      console.warn("Camera not available:", e.message);
    }
  }
}

// ================================================================
// REGISTRATION
// ================================================================
async function registerStudent() {
  const sid = document.getElementById("reg_sid").value.trim();
  if (!sid) return showStatus("status_register", "Enter a student ID", "err");
  try {
    await apiFetch("/register", { method: "POST", body: formData({ student_id: sid }) });
    showStatus("status_register", `✅ ${sid} registered`, "ok");
    document.getElementById("reg_sid").value = "";
  } catch (e) {
    showStatus("status_register", "❌ " + e.message, "err");
  }
}

// ================================================================
// INDIVIDUAL ATTENDANCE
// ================================================================
let individualRunning = false;

async function startIndividual() {
  if (individualRunning) return;
  const sid = document.getElementById("sid_individual").value.trim();
  if (!sid) return showStatus("status_individual", "Enter your student ID", "err");

  individualRunning = true;
  const btn = document.getElementById("btn_start");
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div> Preparing...';
  hideStatus("status_individual");

  const videoEl = document.getElementById("video0");
  try {
    await ensureCamera("video0");
  } catch (e) {
    showStatus("status_individual", "❌ Camera access denied", "err");
    resetIndividualBtn();
    return;
  }

  let challengeData;
  try {
    challengeData = await apiFetch("/challenge", {
      method: "POST",
      body: formData({ student_id: sid }),
    });
  } catch (e) {
    showStatus("status_individual", "❌ " + e.message, "err");
    resetIndividualBtn();
    return;
  }

  const { challenge, challenge_id } = challengeData;
  document.getElementById("challenge_text").textContent = "↔ " + challenge;
  document.getElementById("challenge_hint").textContent = "Move your head " + challenge.toLowerCase() + " now";
  document.getElementById("cameraWrap0").classList.add("detecting");
  showStatus("status_individual", "👁 Detection active — move your head " + challenge.toLowerCase(), "info");
  btn.innerHTML = "Detecting...";

  detectHeadMovement(
    videoEl, challenge,
    async () => {
      // Success callback
      document.getElementById("cameraWrap0").classList.remove("detecting");
      document.getElementById("cameraWrap0").classList.add("detected");
      document.getElementById("challenge_text").textContent = "✓";
      document.getElementById("challenge_hint").textContent = "Movement confirmed!";

      try {
        await apiFetch("/mark_attendance", {
          method: "POST",
          body: formData({ student_id: sid, movement: challenge, challenge_id }),
        });
        showStatus("status_individual", "✅ Attendance marked for " + sid, "ok");
      } catch (e) {
        showStatus("status_individual", "❌ " + e.message, "err");
      }
      setTimeout(() => {
        document.getElementById("cameraWrap0").classList.remove("detected");
        document.getElementById("challenge_text").textContent = "—";
        document.getElementById("challenge_hint").textContent = "Press Start to begin";
        resetIndividualBtn();
      }, 2500);
    },
    () => {
      // Timeout callback
      showStatus("status_individual", "⏱ Timeout — please try again", "err");
      document.getElementById("cameraWrap0").classList.remove("detecting");
      document.getElementById("challenge_text").textContent = "—";
      document.getElementById("challenge_hint").textContent = "Press Start to begin";
      resetIndividualBtn();
    }
  );
}

function resetIndividualBtn() {
  individualRunning = false;
  const btn = document.getElementById("btn_start");
  btn.disabled = false;
  btn.innerHTML = "▶ Start Attendance";
}

// ================================================================
// BATCH ATTENDANCE
// ================================================================
let batchQueue = [];
let batchIdx = 0;
let batchRunning = false;

async function startBatch() {
  if (batchRunning) return;
  const text = document.getElementById("batch_ids").value.trim();
  if (!text) return showStatus("status_batch", "Enter at least one student ID", "err");

  batchQueue = text.split("\n").map(s => s.trim()).filter(Boolean);
  batchIdx = 0;
  batchRunning = true;

  document.getElementById("btn_batch").disabled = true;
  renderBatchList();

  await ensureCamera("video1");
  await processNextBatch();
}

async function processNextBatch() {
  if (batchIdx >= batchQueue.length) {
    showStatus("status_batch", "🎉 Batch complete! " + batchQueue.length + " students processed", "ok");
    document.getElementById("batch_progress").style.width = "100%";
    document.getElementById("btn_batch").disabled = false;
    document.getElementById("batch_challenge_text").textContent = "✓";
    document.getElementById("batch_challenge_hint").textContent = "All done";
    batchRunning = false;
    return;
  }

  const sid = batchQueue[batchIdx];
  const pct = Math.round((batchIdx / batchQueue.length) * 100);
  document.getElementById("batch_progress").style.width = pct + "%";
  updateBatchItem(batchIdx, "active", "active");
  showStatus("status_batch", `Processing ${batchIdx + 1}/${batchQueue.length}: ${sid}`, "info");

  let challengeData;
  try {
    challengeData = await apiFetch("/challenge", {
      method: "POST",
      body: formData({ student_id: sid }),
    });
  } catch (e) {
    updateBatchItem(batchIdx, "pill-fail", "❌ " + e.message);
    batchIdx++;
    setTimeout(processNextBatch, 400);
    return;
  }

  const { challenge, challenge_id } = challengeData;
  document.getElementById("batch_challenge_text").textContent = "↔ " + challenge;
  document.getElementById("batch_challenge_hint").textContent = sid + " → move " + challenge.toLowerCase();

  const videoEl = document.getElementById("video1");

  stopDetection();
  detectHeadMovement(
    videoEl, challenge,
    async () => {
      updateBatchItem(batchIdx, "pill-ok", "✅ Marked");
      try {
        await apiFetch("/mark_attendance", {
          method: "POST",
          body: formData({ student_id: sid, movement: challenge, challenge_id }),
        });
      } catch (e) {
        updateBatchItem(batchIdx, "pill-fail", "❌ " + e.message);
      }
      batchIdx++;
      setTimeout(processNextBatch, 600);
    },
    () => {
      updateBatchItem(batchIdx, "pill-fail", "⏱ Timeout");
      batchIdx++;
      setTimeout(processNextBatch, 400);
    }
  );
}

function renderBatchList() {
  document.getElementById("batch_list").innerHTML = batchQueue.map((id, i) => `
    <div class="batch-item" id="bi_${i}">
      <span>${id}</span>
      <span class="batch-pill pill-pending" id="bp_${i}">Pending</span>
    </div>
  `).join("");
}

function updateBatchItem(i, pillClass, text) {
  const pill = document.getElementById("bp_" + i);
  const row = document.getElementById("bi_" + i);
  if (!pill) return;
  pill.className = "batch-pill " + pillClass;
  pill.textContent = text;
  if (row) {
    row.classList.remove("active");
    if (pillClass === "active") row.classList.add("active");
    row.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

// ================================================================
// ADMIN
// ================================================================
let adminToken = null;

async function adminLogin() {
  const username = document.getElementById("admin_user").value.trim();
  const password = document.getElementById("admin_pass").value;
  if (!username || !password) return showStatus("status_login", "Enter username and password", "err");

  try {
    const data = await apiFetch("/admin/login", {
      method: "POST",
      body: formData({ username, password }),
    });
    adminToken = data.access_token;
    localStorage.setItem("admin_token", adminToken);
    showAdminPanel(username);
    refreshAdmin();
  } catch (e) {
    showStatus("status_login", "❌ " + e.message, "err");
  }
}

function showAdminPanel(username) {
  document.getElementById("login_section").style.display = "none";
  document.getElementById("admin_section").style.display = "block";
  document.getElementById("admin_greeting").textContent =
    "Logged in as " + username + " • Session expires in 2h";
}

function adminLogout() {
  adminToken = null;
  localStorage.removeItem("admin_token");
  document.getElementById("login_section").style.display = "block";
  document.getElementById("admin_section").style.display = "none";
  document.getElementById("admin_user").value = "";
  document.getElementById("admin_pass").value = "";
}

async function refreshAdmin() {
  if (!adminToken) return;
  try {
    const analytics = await apiFetch("/analytics", {
      headers: { Authorization: "Bearer " + adminToken },
    });
    document.getElementById("s_total").textContent = analytics.total_attendance_records;
    document.getElementById("s_today").textContent = analytics.today_count;
    document.getElementById("s_unique").textContent = analytics.unique_students_all_time;
    document.getElementById("admin_greeting").textContent =
      "Logged in as " + analytics.admin + " • Session expires in 2h";

    const students = await apiFetch("/students", {
      headers: { Authorization: "Bearer " + adminToken },
    });
    const tbody = document.getElementById("students_tbody");
    if (!students.length) {
      tbody.innerHTML = '<tr><td colspan="2" style="color:var(--muted);padding:16px;">No students registered yet</td></tr>';
    } else {
      tbody.innerHTML = students.map(s => `
        <tr>
          <td>${s.student_id}</td>
          <td style="color:var(--muted);">${new Date(s.created_at).toLocaleDateString()}</td>
        </tr>
      `).join("");
    }
  } catch (e) {
    if (e.message.includes("401") || e.message.includes("expired")) {
      adminLogout();
    }
  }
}

// ================================================================
// STARTUP
// ================================================================
window.addEventListener("load", async () => {
  try {
    await initMediaPipe();
  } catch (e) {
    console.error("MediaPipe init failed:", e);
    alert("⚠️ Failed to load face detection model. Check your internet connection.");
    return;
  } finally {
    document.getElementById("mp-loading").classList.add("hidden");
  }

  // Start camera for the default tab
  try {
    await startCamera(document.getElementById("video0"));
  } catch (e) {
    console.warn("Initial camera start failed:", e.message);
  }

  // Restore admin session — validate token first
  const saved = localStorage.getItem("admin_token");
  if (saved) {
    adminToken = saved;
    try {
      const check = await apiFetch("/admin/verify-token", {
        method: "POST",
        headers: { Authorization: "Bearer " + saved },
      });
      showAdminPanel(check.username);
      // Don't auto-refresh unless user opens admin tab
    } catch (e) {
      // Token expired or invalid
      localStorage.removeItem("admin_token");
      adminToken = null;
    }
  }
});
</script>
</body>
</html>"""
