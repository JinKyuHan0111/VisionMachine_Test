"""
SQLite DB - 감지 이벤트, 경고 이력, 평가 결과 저장
"""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "detections.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """테이블 생성"""
    with get_connection() as conn:
        conn.executescript("""
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
    print(f"[SQLite] DB 초기화 완료: {DB_PATH}")


def insert_detection(class_name, confidence, source=None, frame_path=None,
                     bbox=None, session_id=None):
    """감지 이벤트 저장 → detection id 반환"""
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
    """경고 이력 저장"""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO alerts
               (detection_id, timestamp, class_name, severity, message)
               VALUES (?, ?, ?, ?, ?)""",
            (detection_id, datetime.now().isoformat(), class_name, severity, message)
        )


def insert_eval_result(model_path, map50, map50_95, precision, recall, f1, fps, notes=None):
    """평가 결과 저장"""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO eval_results
               (timestamp, model_path, map50, map50_95, precision, recall, f1, fps, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), model_path,
             map50, map50_95, precision, recall, f1, fps, notes)
        )


def get_recent_detections(class_name=None, limit=50):
    """최근 감지 이벤트 조회"""
    with get_connection() as conn:
        if class_name:
            rows = conn.execute(
                "SELECT * FROM detections WHERE class_name=? ORDER BY timestamp DESC LIMIT ?",
                (class_name, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def get_recent_alerts(seconds=30, class_name=None):
    """최근 N초 내 경고 조회 (중복 방지용)"""
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
