"""
화재 감지 시스템 웹 대시보드
실행: uvicorn src.web.app:app --reload --port 8000
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.database import sqlite_db

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPTURES_DIR = PROJECT_ROOT / "data" / "captures"

app = FastAPI(title="화재 감지 대시보드")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# 캡처 이미지 정적 파일 서빙
if CAPTURES_DIR.exists():
    app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")


# ─── 페이지 ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# ─── API ───────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    """오늘 / 전체 감지 통계"""
    with sqlite_db.get_connection() as conn:
        today = datetime.now().strftime("%Y-%m-%d")

        total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        today_total = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE timestamp LIKE ?", (f"{today}%",)
        ).fetchone()[0]

        # 클래스별 오늘 감지 수
        class_counts = conn.execute(
            """SELECT class_name, COUNT(*) as cnt
               FROM detections WHERE timestamp LIKE ?
               GROUP BY class_name""", (f"{today}%",)
        ).fetchall()

        # 시간대별 감지 수 (오늘, 시간별)
        hourly = conn.execute(
            """SELECT substr(timestamp, 12, 2) as hour, COUNT(*) as cnt
               FROM detections WHERE timestamp LIKE ?
               GROUP BY hour ORDER BY hour""", (f"{today}%",)
        ).fetchall()

    return {
        "total": total,
        "today": today_total,
        "by_class": {r["class_name"]: r["cnt"] for r in class_counts},
        "hourly": [{"hour": r["hour"], "count": r["cnt"]} for r in hourly],
    }


@app.get("/api/detections")
async def get_detections(limit: int = 50, class_name: str = None):
    """최근 감지 이벤트 목록"""
    rows = sqlite_db.get_recent_detections(class_name=class_name, limit=limit)
    return rows


@app.get("/api/alerts")
async def get_alerts(limit: int = 20):
    """최근 경고 이력"""
    with sqlite_db.get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/captures")
async def get_captures(limit: int = 20):
    """저장된 캡처 이미지 목록"""
    if not CAPTURES_DIR.exists():
        return []
    files = sorted(CAPTURES_DIR.glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [{"filename": f.name, "url": f"/captures/{f.name}"} for f in files[:limit]]


@app.get("/captures/{filename}")
async def serve_capture(filename: str):
    # 경로 조작 방지
    safe_name = Path(filename).name
    file_path = CAPTURES_DIR / safe_name
    if not file_path.exists() or not file_path.resolve().is_relative_to(CAPTURES_DIR.resolve()):
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    return FileResponse(file_path)
