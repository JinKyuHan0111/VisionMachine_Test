# 실시간 영상 화재 감지 알람 시스템

YOLOv8 기반으로 실시간 영상에서 화재를 감지하고 데스크톱 팝업 및 이메일로 즉시 경고를 발송하는 시스템입니다.

## 감지 클래스

| 클래스 | 설명 | 위험 등급 |
|--------|------|-----------|
| `fire` | 화재 (smoke 포함 통합) | CRITICAL |

> smoke로 감지된 결과도 fire로 통합 표시 (모델은 2클래스, 인퍼런스에서 통합)

## 모델 성능 (fire_detector_v2)

| 지표 | 값 |
|------|----|
| mAP@0.5 | 0.809 |
| mAP@0.5:0.95 | 0.688 |
| Precision | 0.910 |
| Recall | 0.671 |
| fire Recall | 0.789 |
| smoke Recall | 0.553 |
| FPS | 28.8 |

> 기본 conf 임계값 0.25 (Recall 우선 — 화재 미탐이 오탐보다 위험)

## 환경

- Python 3.12.2
- NVIDIA RTX 4060 Laptop (CUDA 12.4)
- YOLOv8m (Ultralytics 8.4.26)
- SQLite + ChromaDB

## 빠른 시작

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정 (.env)
```
# 이메일 알림 (Gmail 앱 비밀번호 사용)
EMAIL_SENDER=your_gmail@gmail.com
EMAIL_PASSWORD=앱비밀번호16자리
EMAIL_RECIPIENT=받을이메일@gmail.com

# 웹 대시보드 초기 관리자 계정
DASHBOARD_USER=admin
DASHBOARD_PASSWORD=변경하세요

# 세션 서명 키 (랜덤 문자열로 변경 권장)
SESSION_SECRET_KEY=change-this-to-a-random-secret-key

# 이메일 인증 링크에 사용될 서버 주소
APP_BASE_URL=http://localhost:8000
```

### 3. 학습
```bash
# 처음 학습
python src/training/train.py

# 기존 모델 파인튜닝 (train_config.yaml의 model을 best.pt로 변경 후)
python src/training/train.py
```

### 4. 실시간 감지 (단일 카메라)
```bash
# 동영상 파일
python src/inference/realtime_monitor.py --source 영상파일.mp4

# RTSP 스트림 (CCTV)
python src/inference/realtime_monitor.py --source rtsp://192.168.1.1:554/stream

# 신뢰도 임계값 조정 (기본 0.25)
python src/inference/realtime_monitor.py --source 영상파일.mp4 --conf 0.35
```

### 5. 다중 카메라 모니터링
```bash
# 카메라 2대 예시
python src/inference/multi_monitor.py \
  --sources video1.mp4 video2.mp4 \
  --names "카메라 1" "카메라 2"

# RTSP 스트림
python src/inference/multi_monitor.py \
  --sources rtsp://192.168.1.1:554/stream rtsp://192.168.1.2:554/stream \
  --names "현장 A" "현장 B" \
  --frame-skip 2
```

- 여러 카메라가 한 창에 그리드 형태로 표시됨
- `q` 키로 종료
- `--frame-skip N`: N프레임마다 1번 추론 (카메라가 많을수록 높게 설정)

### 6. 웹 대시보드
```bash
uvicorn src.web.app:app --reload --port 8000
```
브라우저에서 `http://localhost:8000` 접속

#### 계정 등급 체계

| 등급 | 설명 | 접근 범위 |
|------|------|----------|
| 일반 | 회원가입 직후 | 이메일 인증 후 관리자 승인 대기 |
| 회원 | 관리자가 승인 | 담당 카메라 감지 이력 조회 |
| 관리자 | 최고 권한 | 전체 조회 + 사용자 관리 |

#### 회원가입 흐름

```
회원가입 (아이디 / 비밀번호 / 이메일)
  → 인증 이메일 발송
  → 이메일 링크 클릭 (24시간 유효)
  → 관리자 승인 대기
  → 승인 완료 → 대시보드 이용
```

#### 관리자 기능 (`/admin/users`)
- 가입 대기 사용자 승인 / 거절
- 등급 변경 (일반 / 회원 / 관리자)
- 담당 카메라 배정 (쉼표 구분, 비워두면 전체 조회)

### 7. 모델 평가

#### 정량적 평가
```bash
python src/evaluation/evaluate.py --model models/fire_detector_best.pt
```

| 지표 | 설명 | 목표값 |
|------|------|--------|
| **mAP@0.5** | IoU 0.5 기준 평균 정밀도 | > 0.70 |
| **mAP@0.5:0.95** | 다양한 IoU 기준 평균 | > 0.50 |
| **Recall** | 실제 화재 중 감지 비율 | 최우선 (미탐이 더 위험) |
| **Precision** | 감지한 것 중 실제 화재 비율 | 오탐 감소 |
| **FPS** | 초당 처리 프레임 | > 15 |

#### 정성적 평가
```bash
python src/evaluation/visualize_results.py --model models/fire_detector_best.pt
```

### 8. 팝업 경고 테스트
```bash
python src/alert/popup_alert.py
```

## 시스템 구조

```
영상 입력 (파일 / RTSP / 다중 카메라)
    ↓
YOLOv8 추론 (매 프레임, conf=0.25)
    ↓
연속 3프레임 화재 감지
    ↓
┌──────────────┬─────────────────┬───────────┬───────────┐
팝업 경고 발송  이메일 알림 발송   프레임 저장   DB 기록
(화면 우측하단) (Gmail, 5분쿨다운) (captures/) (SQLite)
                                              ↓
                                        웹 대시보드
                                   (계정별 감지 이력 조회)
```

## 프로젝트 구조

```
├── configs/
│   ├── dataset.yaml        # 데이터셋 경로 및 클래스 설정
│   └── train_config.yaml   # 학습 하이퍼파라미터
├── data/
│   ├── raw/                # 학습 데이터셋
│   └── captures/           # 감지 순간 자동 저장 프레임
├── models/                 # 학습된 모델 (.pt)
├── src/
│   ├── alert/              # 팝업 경고 + 이메일 알림
│   ├── data_prep/          # 데이터셋 다운로드 및 병합
│   ├── database/           # SQLite 감지 이력
│   ├── evaluation/         # 정량적 / 정성적 평가
│   ├── inference/          # 실시간 감지 (단일 / 다중 카메라)
│   ├── training/           # YOLOv8 학습
│   └── web/                # FastAPI 웹 대시보드
│       └── templates/      # 로그인, 회원가입, 대시보드 UI
├── runs/                   # 학습 및 평가 결과 (자동 생성)
└── requirements.txt
```

## 데이터셋

학습에 사용한 데이터셋 3종을 병합하여 사용합니다 (총 약 38,000장).

| 데이터셋 | 출처 | 라이선스 | 이미지 수 | 클래스 |
|---------|------|---------|---------|--------|
| smoke-fire-wsde7 v4 | [Roboflow Universe](https://universe.roboflow.com) | CC BY 4.0 | 8,230장 | fire, smoke |
| fire-smoke-mx4z8 v1 | [Roboflow Universe](https://universe.roboflow.com) | CC BY 4.0 | 9,010장 | fire, smoke |
| D-Fire | [gaiasd/DFireDataset](https://github.com/gaiasd/DFireDataset) | MIT | 약 21,000장 | fire, smoke |

> **전처리:** 원본 데이터셋의 `flame` 클래스는 `fire`로 병합하여 2클래스(fire, smoke)로 통일
