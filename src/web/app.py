"""
화재 감지 시스템 웹 대시보드
개발 실행: uvicorn src.web.app:app --reload --port 8000
운영 실행: docker compose up -d  (DEPLOY.md 참고)
"""

import html
import os
import re
import secrets
import smtplib
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bcrypt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from src.database import sqlite_db
from src.database.sqlite_db import ROLE_ADMIN, ROLE_GENERAL, ROLE_MEMBER

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# ─── 설정 ────────────────────────────────────────────────────────

def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


APP_ENV  = os.getenv("APP_ENV", "development").strip().lower()
IS_PROD  = APP_ENV == "production"

_ADMIN_USER     = os.getenv("DASHBOARD_USER", "admin")
_ADMIN_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")
_SESSION_SECRET = os.getenv("SESSION_SECRET_KEY", "")
_EMAIL_SENDER   = os.getenv("EMAIL_SENDER", "")
_EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
_APP_BASE_URL   = os.getenv("APP_BASE_URL", "http://localhost:8000").rstrip("/")
_COOKIE_SECURE  = _env_bool("COOKIE_SECURE", IS_PROD)

# README 예시값 등 절대 운영에 쓰면 안 되는 값
_INSECURE_VALUES = {
    "", "admin", "changeme", "password", "변경하세요",
    "change-this-secret", "change-this-to-a-random-secret-key",
}


def _check_config() -> None:
    """운영 모드에서 위험한 설정이면 서버 기동 자체를 막는다 (fail-fast)."""
    global _SESSION_SECRET, _ADMIN_PASSWORD
    problems = []
    if IS_PROD:
        if _SESSION_SECRET in _INSECURE_VALUES or len(_SESSION_SECRET) < 32:
            problems.append("SESSION_SECRET_KEY 는 32자 이상의 랜덤 문자열이어야 합니다 "
                            "(python -c \"import secrets; print(secrets.token_urlsafe(48))\")")
        if _ADMIN_PASSWORD in _INSECURE_VALUES or len(_ADMIN_PASSWORD) < 12:
            problems.append("DASHBOARD_PASSWORD 는 기본값이 아닌 12자 이상이어야 합니다")
        if not _APP_BASE_URL.startswith("https://"):
            problems.append("APP_BASE_URL 은 https:// 로 시작해야 합니다 (인증 링크/쿠키 보호)")
        if not _COOKIE_SECURE:
            problems.append("운영 모드에서는 COOKIE_SECURE=false 를 사용할 수 없습니다")
        if problems:
            raise RuntimeError("[보안 설정 오류]\n - " + "\n - ".join(problems))
    else:
        if not _SESSION_SECRET:
            _SESSION_SECRET = secrets.token_urlsafe(48)
            print("[경고] SESSION_SECRET_KEY 미설정 — 임시 키 사용 (재시작 시 로그아웃됨)")
        if not _ADMIN_PASSWORD:
            _ADMIN_PASSWORD = "changeme"
            print("[경고] DASHBOARD_PASSWORD 미설정 — 개발용 기본 비밀번호 사용")


_check_config()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPTURES_DIR = PROJECT_ROOT / "data" / "captures"

VALID_ROLES = {ROLE_GENERAL, ROLE_MEMBER, ROLE_ADMIN}
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{3,32}$")
EMAIL_RE    = re.compile(r"^[^@\s<>\"',;]{1,64}@[A-Za-z0-9.\-]{1,190}\.[A-Za-z]{2,24}$")
CAM_SANITIZE_RE = re.compile(r"[^\w\-]")   # inference 의 캡처 파일명 규칙과 동일
PASSWORD_MIN, PASSWORD_MAX_BYTES = 8, 72   # bcrypt 는 72바이트 초과 입력을 거부함
MAX_LIMIT = 200

# 존재하지 않는 계정 로그인 시에도 동일한 시간이 걸리도록 하는 더미 해시
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt())


# ─── 앱 / 미들웨어 ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    sqlite_db.init_db()
    if sqlite_db.get_user(_ADMIN_USER) is None:
        sqlite_db.create_user(
            username=_ADMIN_USER,
            password_hash=bcrypt.hashpw(_ADMIN_PASSWORD.encode(), bcrypt.gensalt()).decode(),
            role=ROLE_ADMIN,
            email_verified=True,
        )
        print(f"[대시보드] 초기 관리자 계정 생성: {_ADMIN_USER}")
    yield


app = FastAPI(
    title="화재 감지 대시보드",
    lifespan=lifespan,
    # 운영에서는 API 문서(/docs, /redoc, /openapi.json) 비공개
    docs_url=None if IS_PROD else "/docs",
    redoc_url=None if IS_PROD else "/redoc",
    openapi_url=None if IS_PROD else "/openapi.json",
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    # 인증 토큰이 URL 에 들어가므로 외부 사이트로 Referer 유출 방지
    # (no-referrer 로 하면 폼 POST 의 Origin 이 null 이 되어 CSRF 검사와 충돌)
    "Referrer-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
        "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
    ),
    "Cache-Control": "no-store",
}

_ALLOWED_ORIGIN_HOSTS = {urlparse(_APP_BASE_URL).netloc}


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # CSRF 방어: 상태 변경 요청은 같은 출처(Origin/Referer)에서 온 것만 허용
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        source = request.headers.get("origin") or request.headers.get("referer")
        if source:
            host = urlparse(source).netloc
            if host not in _ALLOWED_ORIGIN_HOSTS and host != request.headers.get("host"):
                return JSONResponse(status_code=403, content={"detail": "잘못된 요청 출처"})
        elif IS_PROD:
            return JSONResponse(status_code=403, content={"detail": "요청 출처 확인 불가"})

    response = await call_next(request)
    for k, v in _SECURITY_HEADERS.items():
        response.headers.setdefault(k, v)
    return response


# SessionMiddleware 는 마지막에 추가 → 가장 바깥에서 실행됨
app.add_middleware(
    SessionMiddleware,
    secret_key=_SESSION_SECRET,
    session_cookie="vm_session",
    max_age=60 * 60 * 12,        # 12시간
    same_site="lax",
    https_only=_COOKIE_SECURE,   # 운영: HTTPS 에서만 쿠키 전송
)

# 주의: /captures 를 StaticFiles 로 mount 하면 아래 인증 라우트보다 먼저 매칭되어
# 로그인 없이 캡처 이미지가 노출된다. 반드시 serve_capture 라우트로만 제공할 것.


# ─── 요청 제한 (브루트포스 / 메일 폭탄 방지) ─────────────────────

class RateLimiter:
    """단일 프로세스용 슬라이딩 윈도우 제한기 (uvicorn --workers 1 기준)."""

    def __init__(self, max_hits: int, window_sec: int):
        self.max_hits = max_hits
        self.window = window_sec
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str) -> bool:
        """허용되면 True. 기록도 함께 남김."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.max_hits:
                return False
            q.append(now)
            if len(self._hits) > 10_000:   # 메모리 보호
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return True


_login_ip_limiter   = RateLimiter(max_hits=10, window_sec=300)   # IP당 5분 10회
_login_user_limiter = RateLimiter(max_hits=5,  window_sec=300)   # 계정당 5분 5회
_register_limiter   = RateLimiter(max_hits=5,  window_sec=3600)  # IP당 1시간 5회
_email_limiter      = RateLimiter(max_hits=3,  window_sec=600)   # 계정당 10분 3회


def _client_ip(request: Request) -> str:
    # uvicorn --proxy-headers + FORWARDED_ALLOW_IPS 로 신뢰된 프록시의 X-Forwarded-For 만 반영됨
    return request.client.host if request.client else "unknown"


def _clamp_limit(limit: int) -> int:
    return max(1, min(limit, MAX_LIMIT))


# ─── 이메일 인증 발송 ─────────────────────────────────────────────

def _send_verification_email(to_email: str, username: str, token: str) -> bool:
    verify_url = f"{_APP_BASE_URL}/verify-email/{token}"
    if not _EMAIL_SENDER or not _EMAIL_PASSWORD:
        if IS_PROD:
            print("[이메일] 설정 없음 — 인증 메일 미발송")
        else:
            print(f"[이메일] 설정 없음 — 인증 링크: {verify_url}")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "[화재 감지 시스템] 이메일 인증"
    msg["From"]    = _EMAIL_SENDER
    msg["To"]      = to_email

    safe_name = html.escape(username)
    safe_url  = html.escape(verify_url, quote=True)
    body = f"""
    <div style="font-family:sans-serif; max-width:480px; margin:0 auto;">
      <h2 style="color:#ff6b35;">화재 감지 대시보드</h2>
      <p>안녕하세요, <strong>{safe_name}</strong>님.</p>
      <p>아래 버튼을 눌러 이메일 인증을 완료하세요. (24시간 이내)</p>
      <a href="{safe_url}"
         style="display:inline-block; margin:20px 0; padding:12px 28px;
                background:#ff6b35; color:#fff; border-radius:7px;
                text-decoration:none; font-weight:600;">
        이메일 인증하기
      </a>
      <p style="color:#999; font-size:0.85em;">
        버튼이 작동하지 않으면 아래 링크를 복사해 브라우저에 붙여넣으세요.<br>
        {safe_url}
      </p>
    </div>
    """
    msg.attach(MIMEText(body, "html", "utf-8"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.starttls()
            server.login(_EMAIL_SENDER, _EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"[이메일] 발송 실패: {type(e).__name__}")
        return False


# ─── 인증 헬퍼 ───────────────────────────────────────────────────

def get_current_user(request: Request) -> dict:
    username = request.session.get("username")
    if not username:
        raise HTTPException(status_code=307, headers={"Location": "/login"})
    user = sqlite_db.get_user(username)
    if not user:
        request.session.clear()
        raise HTTPException(status_code=307, headers={"Location": "/login"})
    return user


def require_verified(user: dict = Depends(get_current_user)) -> dict:
    """이메일 인증 완료 필요"""
    if not user.get("email_verified"):
        raise HTTPException(status_code=307, headers={"Location": "/verify-pending"})
    return user


def require_member(user: dict = Depends(require_verified)) -> dict:
    """회원(member) 이상 필요"""
    if user["role"] == ROLE_GENERAL:
        raise HTTPException(status_code=307, headers={"Location": "/pending"})
    return user


def require_admin(user: dict = Depends(require_member)) -> dict:
    if user["role"] != ROLE_ADMIN:
        raise HTTPException(status_code=403, detail="관리자 권한 필요")
    return user


def _login_session(request: Request, username: str) -> None:
    """세션 고정 방지: 기존 세션을 비우고 새로 발급"""
    request.session.clear()
    request.session["username"] = username


# ─── 헬스체크 (Docker HEALTHCHECK 용) ────────────────────────────

@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"status": "ok"}


# ─── 회원가입 ────────────────────────────────────────────────────

def _register_error(request: Request, msg: str, code: int = 400):
    return templates.TemplateResponse(request=request, name="register.html",
                                      context={"error": msg, "sent": False},
                                      status_code=code)


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    if request.session.get("username"):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request=request, name="register.html",
                                      context={"error": None, "sent": False})


@app.post("/register", response_class=HTMLResponse)
async def register_post(request: Request,
                        username: str = Form(..., max_length=64),
                        password: str = Form(..., max_length=256),
                        email:    str = Form(..., max_length=254)):
    username, email = username.strip(), email.strip().lower()

    if not USERNAME_RE.fullmatch(username):
        return _register_error(request, "아이디는 영문/숫자/._- 3~32자로 입력하세요.")
    if not EMAIL_RE.fullmatch(email):
        return _register_error(request, "올바른 이메일 주소를 입력하세요.")
    pw_bytes = len(password.encode())
    if len(password) < PASSWORD_MIN or pw_bytes > PASSWORD_MAX_BYTES:
        return _register_error(request, f"비밀번호는 {PASSWORD_MIN}자 이상, "
                                        f"{PASSWORD_MAX_BYTES}바이트 이하로 입력하세요.")
    if not _register_limiter.hit(_client_ip(request)):
        return _register_error(request, "요청이 너무 많습니다. 잠시 후 다시 시도하세요.", 429)

    if sqlite_db.get_user(username):
        return _register_error(request, "이미 사용 중인 아이디입니다.")
    if sqlite_db.get_user_by_email(email):
        return _register_error(request, "이미 사용 중인 이메일입니다.")

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    sqlite_db.create_user(username=username, password_hash=pw_hash,
                          email=email, role=ROLE_GENERAL, email_verified=False)

    token = sqlite_db.create_email_token(username)
    _email_limiter.hit(username)
    _send_verification_email(email, username, token)

    # 로그인 상태로 전환 후 인증 대기 페이지로
    _login_session(request, username)
    return RedirectResponse("/verify-pending", status_code=302)


# ─── 이메일 인증 ─────────────────────────────────────────────────

@app.get("/verify-pending", response_class=HTMLResponse)
async def verify_pending_page(request: Request, user: dict = Depends(get_current_user)):
    if user.get("email_verified"):
        return RedirectResponse("/pending" if user["role"] == ROLE_GENERAL else "/",
                                status_code=302)
    return templates.TemplateResponse(request=request, name="verify_pending.html",
                                      context={"user": user, "resent": False})


@app.post("/verify-pending/resend", response_class=HTMLResponse)
async def resend_verification(request: Request, user: dict = Depends(get_current_user)):
    if user.get("email_verified"):
        return RedirectResponse("/", status_code=302)
    if not user.get("email") or not _email_limiter.hit(user["username"]):
        return templates.TemplateResponse(request=request, name="verify_pending.html",
                                          context={"user": user, "resent": False})
    token = sqlite_db.create_email_token(user["username"])
    _send_verification_email(user["email"], user["username"], token)
    return templates.TemplateResponse(request=request, name="verify_pending.html",
                                      context={"user": user, "resent": True})


@app.get("/verify-email/{token}", response_class=HTMLResponse)
async def verify_email(request: Request, token: str):
    if len(token) > 128:
        return templates.TemplateResponse(request=request, name="verify_result.html",
                                          context={"success": False})
    username = sqlite_db.consume_email_token(token)
    if username is None:
        return templates.TemplateResponse(request=request, name="verify_result.html",
                                          context={"success": False})
    _login_session(request, username)
    return templates.TemplateResponse(request=request, name="verify_result.html",
                                      context={"success": True, "username": username})


# ─── 로그인 / 로그아웃 ───────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("username"):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request=request, name="login.html",
                                      context={"error": None})


@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request,
                     username: str = Form(..., max_length=64),
                     password: str = Form(..., max_length=256)):
    username = username.strip()
    if not _login_ip_limiter.hit(_client_ip(request)) or \
            not _login_user_limiter.hit(username.lower()):
        return templates.TemplateResponse(
            request=request, name="login.html", status_code=429,
            context={"error": "로그인 시도가 너무 많습니다. 5분 후 다시 시도하세요."})

    pw = password.encode()[:PASSWORD_MAX_BYTES]
    user = sqlite_db.get_user(username)
    stored = (user or {}).get("password_hash")
    # 계정 존재 여부와 무관하게 bcrypt 를 수행해 응답 시간으로 계정 유추 방지
    ok = bcrypt.checkpw(pw, stored.encode() if stored else _DUMMY_HASH) and bool(stored)

    if ok:
        _login_session(request, username)
        if not user.get("email_verified"):
            return RedirectResponse("/verify-pending", status_code=302)
        if user["role"] == ROLE_GENERAL:
            return RedirectResponse("/pending", status_code=302)
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request=request, name="login.html", status_code=401,
                                      context={"error": "아이디 또는 비밀번호가 틀렸습니다."})


@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


# ─── 승인 대기 페이지 ────────────────────────────────────────────

@app.get("/pending", response_class=HTMLResponse)
async def pending_page(request: Request, user: dict = Depends(require_verified)):
    if user["role"] != ROLE_GENERAL:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request=request, name="pending.html",
                                      context={"user": user})


# ─── 대시보드 페이지 ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, user: dict = Depends(require_member)):
    return templates.TemplateResponse(request=request, name="index.html",
                                      context={"user": user,
                                               "role_labels": {
                                                   ROLE_GENERAL: "일반",
                                                   ROLE_MEMBER:  "회원",
                                                   ROLE_ADMIN:   "관리자",
                                               }})


# ─── API ─────────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats(user: dict = Depends(require_member)):
    cameras = user["cameras"]
    today = datetime.now().strftime("%Y-%m-%d")

    with sqlite_db.get_connection() as conn:
        cam_cond, cam_params = "", []
        if cameras is not None:
            placeholders = ",".join("?" * len(cameras))
            cam_cond = f" AND source IN ({placeholders})"
            cam_params = cameras

        total = conn.execute(
            f"SELECT COUNT(*) FROM detections WHERE 1=1{cam_cond}", cam_params
        ).fetchone()[0]
        today_total = conn.execute(
            f"SELECT COUNT(*) FROM detections WHERE timestamp LIKE ?{cam_cond}",
            [f"{today}%"] + cam_params
        ).fetchone()[0]
        class_counts = conn.execute(
            f"""SELECT class_name, COUNT(*) as cnt FROM detections
                WHERE timestamp LIKE ?{cam_cond} GROUP BY class_name""",
            [f"{today}%"] + cam_params
        ).fetchall()
        hourly = conn.execute(
            f"""SELECT substr(timestamp,12,2) as hour, COUNT(*) as cnt
                FROM detections WHERE timestamp LIKE ?{cam_cond}
                GROUP BY hour ORDER BY hour""",
            [f"{today}%"] + cam_params
        ).fetchall()

    return {
        "total": total,
        "today": today_total,
        "by_class": {r["class_name"]: r["cnt"] for r in class_counts},
        "hourly": [{"hour": r["hour"], "count": r["cnt"]} for r in hourly],
    }


@app.get("/api/detections")
async def get_detections(limit: int = 50, class_name: str | None = None,
                         user: dict = Depends(require_member)):
    return sqlite_db.get_recent_detections(
        class_name=class_name, limit=_clamp_limit(limit), cameras=user["cameras"]
    )


@app.get("/api/alerts")
async def get_alerts(limit: int = 20, user: dict = Depends(require_member)):
    cameras = user["cameras"]
    limit = _clamp_limit(limit)
    with sqlite_db.get_connection() as conn:
        if cameras is not None:
            placeholders = ",".join("?" * len(cameras))
            rows = conn.execute(
                f"""SELECT a.* FROM alerts a
                    LEFT JOIN detections d ON a.detection_id = d.id
                    WHERE d.source IN ({placeholders})
                    ORDER BY a.timestamp DESC LIMIT ?""",
                cameras + [limit]
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def _capture_allowed(filename: str, cameras: list | None) -> bool:
    """캡처 파일명 {YYYYMMDD}_{HHMMSS}_{camera}_{class}.jpg 의 camera 부분이 담당 카메라인지 확인"""
    if cameras is None:
        return True
    parts = Path(filename).stem.split("_", 2)
    if len(parts) < 3 or "_" not in parts[2]:
        return False
    cam = parts[2].rsplit("_", 1)[0]
    return cam in {CAM_SANITIZE_RE.sub("_", c) for c in cameras}


@app.get("/api/captures")
async def get_captures(limit: int = 20, user: dict = Depends(require_member)):
    if not CAPTURES_DIR.exists():
        return []
    limit = _clamp_limit(limit)
    cameras = user["cameras"]
    files = sorted(CAPTURES_DIR.glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)
    result = []
    for f in files:
        if not _capture_allowed(f.name, cameras):
            continue
        result.append({"filename": f.name, "url": f"/captures/{f.name}"})
        if len(result) >= limit:
            break
    return result


@app.get("/captures/{filename}")
async def serve_capture(filename: str, user: dict = Depends(require_member)):
    safe_name = Path(filename).name
    file_path = CAPTURES_DIR / safe_name
    if (safe_name != filename or file_path.suffix.lower() != ".jpg"
            or not _capture_allowed(safe_name, user["cameras"])
            or not file_path.is_file()
            or not file_path.resolve().is_relative_to(CAPTURES_DIR.resolve())):
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    return FileResponse(file_path, media_type="image/jpeg")


# ─── 관리자 — 사용자 관리 ─────────────────────────────────────────

def _validate_role(role: str) -> str:
    if role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail="알 수 없는 등급입니다.")
    return role


@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users_page(request: Request, admin: dict = Depends(require_admin)):
    pending  = sqlite_db.list_pending_users()
    all_users = [u for u in sqlite_db.list_users() if u["role"] != ROLE_GENERAL]
    return templates.TemplateResponse(request=request, name="admin_users.html",
                                      context={"admin": admin,
                                               "pending": pending,
                                               "users": all_users,
                                               "message": None,
                                               "ROLE_MEMBER": ROLE_MEMBER,
                                               "ROLE_ADMIN": ROLE_ADMIN})


@app.post("/admin/users/approve")
async def admin_approve(username: str = Form(...),
                        role: str = Form(ROLE_MEMBER),
                        cameras: str = Form("", max_length=2000),
                        admin: dict = Depends(require_admin)):
    _validate_role(role)
    cam_list = [c.strip() for c in cameras.split(",") if c.strip()] or None
    sqlite_db.update_user_role(username, role)
    sqlite_db.update_user_cameras(username, cam_list)
    return RedirectResponse("/admin/users", status_code=302)


@app.post("/admin/users/role")
async def admin_change_role(username: str = Form(...),
                             role: str = Form(...),
                             admin: dict = Depends(require_admin)):
    _validate_role(role)
    if username == admin["username"] and role != ROLE_ADMIN:
        raise HTTPException(status_code=400, detail="자기 자신의 관리자 권한은 변경할 수 없습니다.")
    sqlite_db.update_user_role(username, role)
    return RedirectResponse("/admin/users", status_code=302)


@app.post("/admin/users/cameras")
async def admin_update_cameras(username: str = Form(...),
                                cameras: str = Form("", max_length=2000),
                                admin: dict = Depends(require_admin)):
    cam_list = [c.strip() for c in cameras.split(",") if c.strip()] or None
    sqlite_db.update_user_cameras(username, cam_list)
    return RedirectResponse("/admin/users", status_code=302)


@app.post("/admin/users/delete")
async def admin_delete_user(username: str = Form(...),
                             admin: dict = Depends(require_admin)):
    if username == admin["username"]:
        raise HTTPException(status_code=400, detail="자기 자신은 삭제할 수 없습니다.")
    sqlite_db.delete_user(username)
    return RedirectResponse("/admin/users", status_code=302)
