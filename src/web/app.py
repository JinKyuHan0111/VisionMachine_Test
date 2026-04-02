"""
화재 감지 시스템 웹 대시보드
실행: uvicorn src.web.app:app --reload --port 8000
"""

import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bcrypt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from src.database import sqlite_db
from src.database.sqlite_db import ROLE_ADMIN, ROLE_GENERAL, ROLE_MEMBER

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_ADMIN_USER     = os.getenv("DASHBOARD_USER", "admin")
_ADMIN_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "changeme")
_SESSION_SECRET = os.getenv("SESSION_SECRET_KEY", "change-this-secret")
_EMAIL_SENDER   = os.getenv("EMAIL_SENDER", "")
_EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
_APP_BASE_URL   = os.getenv("APP_BASE_URL", "http://localhost:8000").rstrip("/")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPTURES_DIR = PROJECT_ROOT / "data" / "captures"

app = FastAPI(title="화재 감지 대시보드")
app.add_middleware(SessionMiddleware, secret_key=_SESSION_SECRET, max_age=86400)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

if CAPTURES_DIR.exists():
    app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")


# ─── 이메일 인증 발송 ─────────────────────────────────────────────

def _send_verification_email(to_email: str, username: str, token: str) -> bool:
    if not _EMAIL_SENDER or not _EMAIL_PASSWORD:
        print(f"[이메일] 설정 없음 — 인증 링크: {_APP_BASE_URL}/verify-email/{token}")
        return False

    verify_url = f"{_APP_BASE_URL}/verify-email/{token}"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "[화재 감지 시스템] 이메일 인증"
    msg["From"]    = _EMAIL_SENDER
    msg["To"]      = to_email

    html = f"""
    <div style="font-family:sans-serif; max-width:480px; margin:0 auto;">
      <h2 style="color:#ff6b35;">화재 감지 대시보드</h2>
      <p>안녕하세요, <strong>{username}</strong>님.</p>
      <p>아래 버튼을 눌러 이메일 인증을 완료하세요. (24시간 이내)</p>
      <a href="{verify_url}"
         style="display:inline-block; margin:20px 0; padding:12px 28px;
                background:#ff6b35; color:#fff; border-radius:7px;
                text-decoration:none; font-weight:600;">
        이메일 인증하기
      </a>
      <p style="color:#999; font-size:0.85em;">
        버튼이 작동하지 않으면 아래 링크를 복사해 브라우저에 붙여넣으세요.<br>
        {verify_url}
      </p>
    </div>
    """
    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(_EMAIL_SENDER, _EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"[이메일] 발송 실패: {e}")
        return False


# ─── 시작 시 초기 관리자 계정 생성 ───────────────────────────────

@app.on_event("startup")
def startup():
    sqlite_db.init_db()
    if sqlite_db.get_user(_ADMIN_USER) is None:
        sqlite_db.create_user(
            username=_ADMIN_USER,
            password_hash=bcrypt.hashpw(_ADMIN_PASSWORD.encode(), bcrypt.gensalt()).decode(),
            role=ROLE_ADMIN,
            email_verified=True,
        )
        print(f"[대시보드] 초기 관리자 계정 생성: {_ADMIN_USER}")


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


# ─── 회원가입 ────────────────────────────────────────────────────

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    if request.session.get("username"):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request=request, name="register.html",
                                      context={"error": None, "sent": False})


@app.post("/register", response_class=HTMLResponse)
async def register_post(request: Request,
                        username: str = Form(...),
                        password: str = Form(...),
                        email:    str = Form(...)):
    if sqlite_db.get_user(username):
        return templates.TemplateResponse(request=request, name="register.html",
                                          context={"error": "이미 사용 중인 아이디입니다.", "sent": False})
    if sqlite_db.get_user_by_email(email):
        return templates.TemplateResponse(request=request, name="register.html",
                                          context={"error": "이미 사용 중인 이메일입니다.", "sent": False})

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    sqlite_db.create_user(username=username, password_hash=pw_hash,
                          email=email, role=ROLE_GENERAL, email_verified=False)

    token = sqlite_db.create_email_token(username)
    _send_verification_email(email, username, token)

    # 로그인 상태로 전환 후 인증 대기 페이지로
    request.session["username"] = username
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
    if not user.get("email"):
        return templates.TemplateResponse(request=request, name="verify_pending.html",
                                          context={"user": user, "resent": False})
    token = sqlite_db.create_email_token(user["username"])
    _send_verification_email(user["email"], user["username"], token)
    return templates.TemplateResponse(request=request, name="verify_pending.html",
                                      context={"user": user, "resent": True})


@app.get("/verify-email/{token}", response_class=HTMLResponse)
async def verify_email(request: Request, token: str):
    username = sqlite_db.consume_email_token(token)
    if username is None:
        return templates.TemplateResponse(request=request, name="verify_result.html",
                                          context={"success": False})
    # 세션 갱신
    request.session["username"] = username
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
                     username: str = Form(...),
                     password: str = Form(...)):
    user = sqlite_db.get_user(username)
    if user and user.get("password_hash") and \
            bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        request.session["username"] = username
        if not user.get("email_verified"):
            return RedirectResponse("/verify-pending", status_code=302)
        if user["role"] == ROLE_GENERAL:
            return RedirectResponse("/pending", status_code=302)
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request=request, name="login.html",
                                      context={"error": "아이디 또는 비밀번호가 틀렸습니다."})


@app.get("/logout")
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
async def get_detections(limit: int = 50, class_name: str = None,
                         user: dict = Depends(require_member)):
    return sqlite_db.get_recent_detections(
        class_name=class_name, limit=limit, cameras=user["cameras"]
    )


@app.get("/api/alerts")
async def get_alerts(limit: int = 20, user: dict = Depends(require_member)):
    cameras = user["cameras"]
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


@app.get("/api/captures")
async def get_captures(limit: int = 20, user: dict = Depends(require_member)):
    if not CAPTURES_DIR.exists():
        return []
    cameras = user["cameras"]
    files = sorted(CAPTURES_DIR.glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)
    result = []
    for f in files:
        if cameras is not None:
            if not any(c.replace(" ", "_") in f.name for c in cameras):
                continue
        result.append({"filename": f.name, "url": f"/captures/{f.name}"})
        if len(result) >= limit:
            break
    return result


@app.get("/captures/{filename}")
async def serve_capture(filename: str, user: dict = Depends(require_member)):
    safe_name = Path(filename).name
    file_path = CAPTURES_DIR / safe_name
    if not file_path.exists() or not file_path.resolve().is_relative_to(CAPTURES_DIR.resolve()):
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    return FileResponse(file_path)


# ─── 관리자 — 사용자 관리 ─────────────────────────────────────────

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
                        cameras: str = Form(""),
                        admin: dict = Depends(require_admin)):
    cam_list = [c.strip() for c in cameras.split(",") if c.strip()] or None
    sqlite_db.update_user_role(username, role)
    sqlite_db.update_user_cameras(username, cam_list)
    return RedirectResponse("/admin/users", status_code=302)


@app.post("/admin/users/role")
async def admin_change_role(username: str = Form(...),
                             role: str = Form(...),
                             admin: dict = Depends(require_admin)):
    if username == admin["username"] and role != ROLE_ADMIN:
        raise HTTPException(status_code=400, detail="자기 자신의 관리자 권한은 변경할 수 없습니다.")
    sqlite_db.update_user_role(username, role)
    return RedirectResponse("/admin/users", status_code=302)


@app.post("/admin/users/cameras")
async def admin_update_cameras(username: str = Form(...),
                                cameras: str = Form(""),
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
