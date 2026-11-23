# 원격 서버 Docker 배포 가이드 (웹 대시보드)

배포 관련 파일은 모두 [docker/](docker/) 폴더에 있습니다.

| 파일 | 역할 |
|---|---|
| `docker/Dockerfile` | 웹 대시보드 이미지 (빌드 컨텍스트는 프로젝트 루트) |
| `docker/Dockerfile.dockerignore` | 이미지에서 제외할 파일 (`.env`, 데이터, 모델 등) |
| `docker/docker-compose.yml` | app + Caddy 구성 |
| `docker/Caddyfile` | HTTPS 리버스 프록시 설정 |
| `docker/.env.example` | 배포용 환경변수 예시 → `docker/.env`로 복사 |

```
인터넷 ──▶ :80/:443  Caddy (자동 HTTPS, HSTS)
                       │  내부 Docker 네트워크 (172.28.0.0/24)
                       ▼
                     app:8000  FastAPI + SQLite   ← 호스트 포트 미공개
                       │
                     [volume] app-data → /app/data (db/, captures/)
```

배포 대상은 **웹 대시보드**뿐입니다. 실시간 감지(`src/inference`)는 GPU와 화면(OpenCV 창, tkinter 팝업)이 필요해서 이 이미지에 넣지 않았습니다.
원격 서버에서는 감지 데이터 없이 빈 대시보드가 뜨는 게 정상입니다.

---

## 1. 서버 준비 (Ubuntu 22.04/24.04 기준)

### 1-1. SSH 보안
```bash
# 로컬 PC에서: 키 생성 후 서버에 등록
ssh-keygen -t ed25519
ssh-copy-id <user>@<서버IP>

# 서버에서: /etc/ssh/sshd_config (또는 /etc/ssh/sshd_config.d/99-hardening.conf)
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no

sudo systemctl restart ssh   # ← 키 로그인이 되는지 새 터미널에서 먼저 확인한 뒤 실행
```

### 1-2. 방화벽
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 443/udp
sudo ufw enable
```
> ⚠️ **Docker는 ufw를 우회합니다.** `ports:`로 공개한 포트는 ufw 규칙과 상관없이 외부에 열립니다.
> 그래서 `docker/docker-compose.yml`에서 app은 `expose`(내부 전용)만 쓰고, 호스트 포트는 Caddy의 80/443만 공개합니다.
> 디버깅한다고 `ports: ["8000:8000"]`을 추가하지 마세요. 꼭 필요하면 `"127.0.0.1:8000:8000"`으로 쓰고 SSH 터널로 접속하세요.

클라우드(AWS/GCP/Oracle 등)를 쓴다면 **보안 그룹/방화벽 규칙**에서도 22, 80, 443만 허용하세요. 22번은 가능하면 내 IP만 허용하는 게 좋습니다.

### 1-3. 자동 보안 업데이트 + fail2ban
```bash
sudo apt update && sudo apt install -y unattended-upgrades fail2ban
sudo dpkg-reconfigure -plow unattended-upgrades
```

### 1-4. Docker 설치
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
less get-docker.sh          # 내용을 확인한 뒤 실행
sudo sh get-docker.sh
```
> `docker` 그룹은 사실상 **root 권한**과 같습니다. 배포 계정 외에는 추가하지 마세요.

---

## 2. 도메인

Caddy가 Let's Encrypt 인증서를 자동으로 발급하려면 도메인이 필요합니다.
- 도메인이 있으면 A 레코드를 서버 IP로 지정합니다.
- 연습용이라면 `<서버IP>.sslip.io`를 쓰면 됩니다(예: `203.0.113.10.sslip.io`). 별도 설정 없이 해당 IP로 연결됩니다.

---

## 3. 배포

```bash
git clone <저장소 URL> && cd VisionMachine_Test/docker   # 이후 명령은 모두 docker/ 에서 실행

cp .env.example .env
chmod 600 .env
python3 -c "import secrets; print(secrets.token_urlsafe(48))"   # → SESSION_SECRET_KEY
nano .env      # DOMAIN, APP_BASE_URL, SESSION_SECRET_KEY, DASHBOARD_PASSWORD, EMAIL_* 입력

docker compose up -d --build
docker compose ps            # app 이 (healthy) 인지 확인
docker compose logs -f app
```

`.env`가 안전하지 않으면 **app 컨테이너가 기동을 거부합니다**(`APP_ENV=production`). 거부 조건은 다음과 같습니다.
- `SESSION_SECRET_KEY`가 32자 미만이거나 README 예시값임
- `DASHBOARD_PASSWORD`가 12자 미만이거나 `changeme` 같은 기본값임
- `APP_BASE_URL`이 `https://`로 시작하지 않음

---

## 4. 배포 후 점검

```bash
# 보안 헤더 확인 (HSTS, CSP, X-Frame-Options 등)
curl -sI https://<도메인>/login

# 앱 포트가 외부에서 닫혀 있는지 (로컬 PC에서) → 연결 실패가 정상
curl -m 5 http://<서버IP>:8000/healthz

# API 문서 비공개 확인 → 404 가 정상
curl -s -o /dev/null -w "%{http_code}\n" https://<도메인>/docs
```

- [ ] 관리자 로그인 확인
- [ ] 첫 로그인 후 `.env`의 `DASHBOARD_PASSWORD`를 지워도 됩니다(계정은 최초 1회만 생성됨)
- [ ] 회원가입 → 인증 메일 링크가 `https://<도메인>/verify-email/...`로 오는지 확인

---

## 5. 백업 / 업데이트

```bash
# DB 백업 (SQLite 온라인 백업)
docker compose exec app python -c "import sqlite3; s=sqlite3.connect('data/db/detections.db'); d=sqlite3.connect('/tmp/b.db'); s.backup(d); d.close()"
docker compose cp app:/tmp/b.db ./backup-$(date +%F).db
chmod 600 backup-*.db         # 비밀번호 해시·이메일 포함

# 코드 / 베이스 이미지 업데이트 (docker/ 에서)
git pull
docker compose build --pull
docker compose up -d

# 의존성 취약점 점검 (로컬에서 주기적으로)
pip install pip-audit && pip-audit -r ../requirements-web.txt
```

---

## 6. 적용된 보안 조치 요약

| 영역 | 조치 |
|---|---|
| 비밀값 | `docker/.env`는 git과 이미지에 들어가지 않음(`.gitignore`, `Dockerfile.dockerignore`). 런타임에 `env_file`로 주입. 기본값이면 기동 거부 |
| 전송 | Caddy 자동 HTTPS + HSTS. 세션 쿠키 `Secure`, `HttpOnly`, `SameSite=Lax` |
| 인증 | 로그인 시도 제한(IP당 5분 10회, 계정당 5분 5회). 가입·메일 재발송 제한. 세션 고정 방지. 응답 시간으로 계정 존재를 알 수 없게 처리 |
| 인가 | 캡처 이미지 **인증 우회 수정**(StaticFiles mount 제거). 담당 카메라가 아닌 캡처는 직접 URL로도 차단. 등급 값 검증 |
| CSRF | 상태를 바꾸는 요청은 Origin/Referer 검사. 로그아웃은 POST만 허용 |
| XSS | 관리자 페이지의 사용자명 JS 삽입 수정. 대시보드 `innerHTML` 이스케이프. CSP, X-Frame-Options |
| 입력 | 아이디·이메일 형식 검증(메일 헤더 인젝션 차단). 비밀번호 8자 이상 72바이트 이하. 조회 `limit` 상한 200 |
| 노출 | 운영 모드에서 `/docs`, `/redoc`, `/openapi.json` 비활성화. `Server` 헤더 제거. Caddy admin API 비활성화 |
| 컨테이너 | 비루트(UID 10001), 읽기 전용 루트 FS, `cap_drop: ALL`, `no-new-privileges`, 메모리/PID/CPU 제한, 로그 로테이션 |
| 네트워크 | app 포트는 비공개. 고정 서브넷의 프록시가 보낸 `X-Forwarded-For`만 신뢰 |

## 7. 남은 리스크 (알고 넘어갈 것)

- **회원가입 시 아이디/이메일 중복 메시지**로 계정 존재 여부를 알 수 있습니다(UX를 위해 유지).
- **요청 제한이 메모리 기반**이라 재시작하면 초기화되고, 워커가 여러 개면 동작하지 않습니다. 그래서 `--workers 1`로 고정했습니다.
- **비밀번호 변경·재설정 기능이 없습니다.** 관리자 비밀번호를 잊으면 DB를 직접 수정해야 합니다.
- CSP에 `'unsafe-inline'`이 들어가 있습니다. 템플릿의 인라인 스크립트와 `onclick`을 외부 `.js`로 분리하면 제거할 수 있습니다.
- 감지 모듈을 원격에서 RTSP로 돌린다면 **카메라 계정/비밀번호가 URL에 들어가 프로세스 목록과 로그에 남습니다.** 환경변수로 넘기고 카메라는 VPN 안에 두세요.
