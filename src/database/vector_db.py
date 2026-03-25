"""
ChromaDB - 감지 이벤트 임베딩 저장 및 유사 사례 검색
텍스트 임베딩 기반 (감지 설명문 → semantic search)
"""

import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from pathlib import Path

CHROMA_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "chroma"

# 클래스별 위험 등급
SEVERITY = {
    "fire":   "CRITICAL",
    "flame":  "CRITICAL",
    "smoke":  "WARNING",
}


def _get_client():
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def _get_collection():
    client = _get_client()
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name="detection_events",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_detection_event(detection_id: int, class_name: str, confidence: float,
                        source: str = None, timestamp: str = None):
    """
    감지 이벤트를 자연어 설명문으로 변환 후 임베딩 저장
    예: "fire detected with 0.92 confidence on camera highway_cam1 at 14:30:15"
    """
    timestamp = timestamp or datetime.now().isoformat()
    source = source or "unknown"
    severity = SEVERITY.get(class_name, "INFO")

    description = (
        f"{class_name} detected with {confidence:.2f} confidence "
        f"on {source} at {timestamp}. "
        f"severity: {severity}."
    )

    collection = _get_collection()
    collection.add(
        ids=[str(detection_id)],
        documents=[description],
        metadatas=[{
            "detection_id": detection_id,
            "class_name":   class_name,
            "confidence":   confidence,
            "source":       source,
            "timestamp":    timestamp,
            "severity":     severity,
        }]
    )


def search_similar(query: str, n_results: int = 5, class_name: str = None):
    """
    자연어 쿼리로 유사 감지 사례 검색
    예: "nighttime fire on highway"
        "high confidence smoke detection"
    """
    collection = _get_collection()
    where = {"class_name": class_name} if class_name else None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "description": doc,
            "similarity":  round(1 - dist, 4),
            **meta,
        })
    return hits


def is_duplicate_alert(class_name: str, source: str, window_seconds: int = 30):
    """
    최근 window_seconds 내 동일 소스에서 같은 클래스 감지 여부 확인
    True → 중복 경고 → 팝업 생략
    """
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(seconds=window_seconds)).isoformat()

    collection = _get_collection()
    results = collection.query(
        query_texts=[f"{class_name} detected on {source}"],
        n_results=5,
        where={"class_name": class_name},
    )

    for meta in results["metadatas"][0]:
        if meta.get("source") == source and meta.get("timestamp", "") > cutoff:
            return True
    return False


def get_stats():
    """저장된 이벤트 수 요약"""
    collection = _get_collection()
    total = collection.count()
    print(f"[ChromaDB] 저장된 감지 이벤트: {total}건")
    return total
