"""
DB Manager - SQLite + ChromaDB 통합 인터페이스
감지 이벤트 저장 시 두 DB에 동시 기록
"""

from src.database import sqlite_db, vector_db

# 클래스별 위험 등급 및 경고 메시지
SEVERITY_MAP = {
    "fire":   ("CRITICAL", "화재 감지! 즉시 확인 필요"),
    "flame":  ("CRITICAL", "화염 감지! 즉시 확인 필요"),
    "smoke":  ("WARNING",  "연기 감지. 주의 요망"),
}


def initialize():
    """SQLite 테이블 생성 + ChromaDB 컬렉션 초기화"""
    sqlite_db.init_db()
    vector_db.get_stats()
    print("[DB Manager] 초기화 완료")


def record_detection(class_name: str, confidence: float,
                     source: str = None, frame_path: str = None,
                     bbox: tuple = None, session_id: str = None):
    """
    감지 이벤트를 SQLite + ChromaDB에 동시 저장
    경고 대상이면 alerts 테이블에도 기록
    Returns: (detection_id, should_alert, severity)
    """
    severity, message = SEVERITY_MAP.get(class_name, ("INFO", f"{class_name} 감지"))

    # 1. 중복 체크 (ChromaDB 추가 전에 먼저 확인)
    is_dup = vector_db.is_duplicate_alert(class_name, source or "unknown", window_seconds=30)
    should_alert = not is_dup

    # 2. SQLite 저장
    detection_id = sqlite_db.insert_detection(
        class_name=class_name,
        confidence=confidence,
        source=source,
        frame_path=frame_path,
        bbox=bbox,
        session_id=session_id,
    )

    # 3. ChromaDB 저장
    vector_db.add_detection_event(
        detection_id=detection_id,
        class_name=class_name,
        confidence=confidence,
        source=source,
    )

    if should_alert:
        sqlite_db.insert_alert(
            detection_id=detection_id,
            class_name=class_name,
            severity=severity,
            message=message,
        )

    return detection_id, should_alert, severity


def search_similar_events(query: str, n_results: int = 5, class_name: str = None):
    """자연어로 유사 사례 검색"""
    return vector_db.search_similar(query, n_results=n_results, class_name=class_name)


def get_recent_detections(class_name: str = None, limit: int = 50):
    """최근 감지 이벤트 조회"""
    return sqlite_db.get_recent_detections(class_name=class_name, limit=limit)


def save_eval_result(model_path, map50, map50_95, precision, recall, f1, fps, notes=None):
    """평가 결과 저장"""
    sqlite_db.insert_eval_result(
        model_path=model_path,
        map50=map50,
        map50_95=map50_95,
        precision=precision,
        recall=recall,
        f1=f1,
        fps=fps,
        notes=notes,
    )
