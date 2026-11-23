# syntax=docker/dockerfile:1
# 화재 감지 웹 대시보드 이미지
# - slim 베이스, 비루트 사용자, 비밀값 미포함(.env 는 런타임에 주입)

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ENV=production

WORKDIR /app

# 고정 UID/GID 의 비루트 사용자 (로그인 셸 없음)
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# 웹 대시보드에 필요한 코드만 복사 (코드는 root 소유 → 앱 프로세스가 수정 불가)
COPY src/__init__.py src/__init__.py
COPY src/database/__init__.py src/database/sqlite_db.py src/database/
COPY src/web/ src/web/

# 앱이 쓰기 가능한 곳은 data/ 뿐 (named volume 으로 마운트됨)
RUN mkdir -p data/db data/captures && chown -R app:app data

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)" || exit 1

# --workers 1: SQLite + 메모리 기반 요청 제한기 사용
# --proxy-headers: FORWARDED_ALLOW_IPS(compose 에서 설정)에 속한 프록시의 X-Forwarded-* 만 신뢰
CMD ["uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--no-server-header"]
