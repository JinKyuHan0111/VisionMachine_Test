# 실시간 영상 화재 감지 알람 시스템

YOLOv8 기반으로 실시간 영상에서 화재와 연기를 감지하고 데스크톱 팝업 및 이메일로 즉시 경고를 발송하는 시스템입니다.

## 감지 클래스

| 클래스 | 설명 | 위험 등급 |
|--------|------|-----------|
| `fire` | 화재 | CRITICAL |
| `smoke` | 연기 | WARNING |

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

### 2. 이메일 알림 설정 (.env)
```
EMAIL_SENDER=your_gmail@gmail.com
EMAIL_PASSWORD=앱비밀번호16자리
EMAIL_RECIPIENT=받을이메일@gmail.com
```

### 3. 학습
```bash
# 처음 학습
python src/training/train.py

# 기존 모델 파인튜닝
python src/training/train.py  # train_config.yaml의 model을 best.pt로 변경 후
```

### 4. 실시간 감지
```bash
# 동영상 파일
python src/inference/realtime_monitor.py --source 영상파일.mp4

# RTSP 스트림 (CCTV)
python src/inference/realtime_monitor.py --source rtsp://192.168.1.1:554/stream

# 신뢰도 임계값 조정 (기본 0.25)
python src/inference/realtime_monitor.py --source 영상파일.mp4 --conf 0.35
```

### 5. 웹 대시보드
```bash
uvicorn src.web.app:app --reload --port 8000
```
브라우저에서 `http://localhost:8000` 접속

### 6. 모델 평가

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

| 출력 | 내용 |
|------|------|
| `tp/` | 정확히 감지된 케이스 |
| `fp/` | 오탐 케이스 (터널 조명, 안개 등) |
| `fn/` | 미탐 케이스 (원거리, 야간, 역광) |
| `confusion_matrix.png` | fire / smoke 혼동 행렬 |
| `pr_curve.png` | 클래스별 Precision-Recall 곡선 |

### 7. 팝업 경고 테스트
```bash
python src/alert/popup_alert.py
```

## 시스템 구조

```
영상 입력 (파일 / RTSP)
    ↓
YOLOv8 추론 (매 프레임, conf=0.25)
    ↓
연속 3프레임 화재/연기 감지
    ↓
┌──────────────┬─────────────────┬───────────┬───────────┐
팝업 경고 발송  이메일 알림 발송   프레임 저장   DB 기록
(화면 우측하단) (Gmail, 5분쿨다운) (captures/) (SQLite+ChromaDB)
```

## 프로젝트 구조

```
├── configs/
│   ├── dataset.yaml        # 데이터셋 경로 및 클래스 설정
│   └── train_config.yaml   # 학습 하이퍼파라미터
├── data/
│   ├── raw/                # 학습 데이터셋 (fire, smoke)
│   └── captures/           # 감지 순간 자동 저장 프레임
├── models/                 # 학습된 모델 (.pt)
├── src/
│   ├── alert/              # 팝업 경고 + 이메일 알림
│   ├── data_prep/          # 데이터셋 다운로드 및 병합
│   ├── database/           # SQLite + ChromaDB 감지 이력
│   ├── evaluation/         # 정량적 / 정성적 평가
│   ├── inference/          # 실시간 감지 및 추론
│   ├── training/           # YOLOv8 학습
│   └── web/                # FastAPI 웹 대시보드
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
