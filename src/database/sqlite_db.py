"""
SQLite DB - 감지 이벤트, 경고 이력, 평가 결과 저장
"""

import json
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "detections.db"

# 역할 상수
ROLE_GENERAL = "general"   # 일반 — 가입 직후, 대시보드 접근 불가
ROLE_MEMBER  = "member"    # 회원 — 관리자가 승인, 담당 카메라 조회 가능
ROLE_ADMIN   = "admin"     # 관리자 — 전체 조회 + 사용자 관리


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_users(conn):
    """기존 users 테이블에 새 컬럼 추가 (하위 호환)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    additions = [
        ("email",          "TEXT"),
        ("google_id",      "TEXT"),
        ("role",           "TEXT DEFAULT 'general'"),
        ("email_verified", "INTEGER DEFAULT 0"),
    ]
    for col, col_type in additions:
        if col not in existing:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")

    # 기존 is_admin=1 계정 → role=admin, email_verified=1 마이그레이션
    if "is_admin" in existing:
        conn.execute(
            "UPDATE users SET role = 'admin', email_verified = 1 "
            "WHERE is_admin = 1 AND (role IS NULL OR role = 'general')"
        )
    # 기존에 role=admin/member 인데 email_verified 미설정인 경우 인증 처리
    conn.execute(
        "UPDATE users SET email_verified = 1 WHERE role IN ('admin', 'member') AND email_verified = 0"
    )


def init_db():
    """테이블 생성 및 마이그레이션"""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                username       TEXT    UNIQUE NOT NULL,
                email          TEXT    UNIQUE,
                password_hash  TEXT,
                cameras        TEXT,
                role           TEXT    DEFAULT 'general',
                email_verified INTEGER DEFAULT 0,
                created_at     TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS email_tokens (
                token      TEXT PRIMARY KEY,
                username   TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                class_name  TEXT    NOT NULL,
                confidence  REAL    NOT NULL,
                source      TEXT,
                frame_path  TEXT,
                bbox_x1     REAL,
                bbox_y1     REAL,
                bbox_x2     REAL,
                bbox_y2     REAL,
                session_id  TEXT
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER REFERENCES detections(id),
                timestamp    TEXT    NOT NULL,
                class_name   TEXT    NOT NULL,
                severity     TEXT    NOT NULL,
                message      TEXT,
                acknowledged INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS eval_results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                model_path  TEXT,
                map50       REAL,
                map50_95    REAL,
                precision   REAL,
                recall      REAL,
                f1          REAL,
                fps         REAL,
                notes       TEXT
            );
        """)
        _migrate_users(conn)
    print(f"[SQLite] DB 초기화 완료: {DB_PATH}")


# ─── 사용자 관리 ─────────────────────────────────────────────────

def _parse_user(row) -> dict:
    if row is None:
        return None
    u = dict(row)
    u["cameras"] = json.loads(u["cameras"]) if u.get("cameras") else None
    return u


def create_user(username: str, password_hash: str = None, email: str = None,
                cameras: list = None, role: str = ROLE_GENERAL,
                email_verified: bool = False) -> None:
    cameras_json = json.dumps(cameras) if cameras is not None else None
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO users
               (username, email, password_hash, cameras, role, email_verified, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (username, email, password_hash,
             cameras_json, role, int(email_verified), datetime.now().isoformat())
        )


def get_user(username: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return _parse_user(row)


def get_user_by_email(email: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
    return _parse_user(row)


def set_email_verified(username: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE users SET email_verified = 1 WHERE username = ?", (username,))


def list_users() -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, username, email, cameras, role, created_at FROM users ORDER BY id"
        ).fetchall()
    return [_parse_user(r) for r in rows]


def list_pending_users() -> list:
    """승인 대기 중인 일반 사용자 목록"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, username, email, cameras, role, created_at FROM users "
            "WHERE role = ? ORDER BY created_at", (ROLE_GENERAL,)
        ).fetchall()
    return [_parse_user(r) for r in rows]


def update_user_role(username: str, role: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE users SET role = ? WHERE username = ?", (role, username))


def update_user_cameras(username: str, cameras: list | None) -> None:
    cameras_json = json.dumps(cameras) if cameras is not None else None
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET cameras = ? WHERE username = ?", (cameras_json, username)
        )


def create_email_token(username: str) -> str:
    """이메일 인증 토큰 생성 (24시간 유효). 기존 토큰은 삭제."""
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now() + timedelta(hours=24)).isoformat()
    with get_connection() as conn:
        conn.execute("DELETE FROM email_tokens WHERE username = ?", (username,))
        conn.execute(
            "INSERT INTO email_tokens (token, username, expires_at) VALUES (?, ?, ?)",
            (token, username, expires_at)
        )
    return token


def consume_email_token(token: str) -> str | None:
    """토큰 검증 후 username 반환. 만료/없으면 None. 사용된 토큰은 즉시 삭제."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM email_tokens WHERE token = ?", (token,)
        ).fetchone()
        if row is None:
            return None
        if datetime.fromisoformat(row["expires_at"]) < datetime.now():
            conn.execute("DELETE FROM email_tokens WHERE token = ?", (token,))
            return None
        username = row["username"]
        conn.execute("UPDATE users SET email_verified = 1 WHERE username = ?", (username,))
        conn.execute("DELETE FROM email_tokens WHERE token = ?", (token,))
        return username


def delete_user(username: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", (username,))


# ─── 감지 이벤트 ──────────────────────────────────────────────────

def insert_detection(class_name, confidence, source=None, frame_path=None,
                     bbox=None, session_id=None):
    bbox = bbox or (None, None, None, None)
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO detections
               (timestamp, class_name, confidence, source, frame_path,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), class_name, confidence,
             source, frame_path, *bbox, session_id)
        )
        return cur.lastrowid


def insert_alert(detection_id, class_name, severity, message=None):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO alerts
               (detection_id, timestamp, class_name, severity, message)
               VALUES (?, ?, ?, ?, ?)""",
            (detection_id, datetime.now().isoformat(), class_name, severity, message)
        )


def insert_eval_result(model_path, map50, map50_95, precision, recall, f1, fps, notes=None):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO eval_results
               (timestamp, model_path, map50, map50_95, precision, recall, f1, fps, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), model_path,
             map50, map50_95, precision, recall, f1, fps, notes)
        )


def get_recent_detections(class_name=None, limit=50, cameras: list = None):
    with get_connection() as conn:
        conditions = []
        params: list = []

        if class_name:
            conditions.append("class_name = ?")
            params.append(class_name)
        if cameras is not None:
            placeholders = ",".join("?" * len(cameras))
            conditions.append(f"source IN ({placeholders})")
            params.extend(cameras)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM detections {where} ORDER BY timestamp DESC LIMIT ?",
            params
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_alerts(seconds=30, class_name=None):
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(seconds=seconds)).isoformat()
    with get_connection() as conn:
        if class_name:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE timestamp > ? AND class_name=? ORDER BY timestamp DESC",
                (cutoff, class_name)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE timestamp > ? ORDER BY timestamp DESC",
                (cutoff,)
            ).fetchall()
    return [dict(r) for r in rows]
