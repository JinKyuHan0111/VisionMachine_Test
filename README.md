# VisionMachine_Test
고속도로 CCTV 영상 기반 사고/산불 조기 경보 시스템 (YOLOv8)

## 프로젝트 개요
CCTV 영상에서 차량 사고, 화재/산불, 연기, 갓길 정차 차량, 낙하물을 실시간으로 감지하고 데스크톱 팝업으로 경고를 발송합니다.

## 감지 클래스
| 클래스 | 설명 | 위험 등급 |
|--------|------|-----------|
| `accident` | 차량 사고 | CRITICAL |
| `fire` | 화재/산불 | CRITICAL |
| `smoke` | 연기 | WARNING |
| `stopped_vehicle` | 갓길 정차 차량 | INFO |
| `debris` | 낙하물/장애물 | WARNING |

## 환경
- Python 3.12.2
- CUDA 지원 GPU
- YOLOv8 (Ultralytics)

## 빠른 시작

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터셋 다운로드
```bash
python src/data_prep/download_datasets.py
```
안내에 따라 Roboflow Universe 또는 D-Fire 데이터셋 다운로드

### 3. 데이터셋 준비
```bash
# 영상에서 프레임 추출 (CCTV 영상이 있는 경우)
python src/data_prep/extract_frames.py --video_dir data/raw --output data/images/raw_frames

# train/val/test 분리
python src/data_prep/prepare_dataset.py --images data/images --labels data/labels
```

### 4. 학습
```bash
python src/training/train.py
# 학습 재개
python src/training/train.py --resume runs/train/highway_detector/weights/last.pt
```

### 5. 실시간 모니터링
```bash
# 동영상 파일
python src/inference/realtime_monitor.py --source data/raw/highway.mp4 --model models/highway_detector_best.pt

# 웹캠
python src/inference/realtime_monitor.py --source 0

# RTSP 스트림
python src/inference/realtime_monitor.py --source rtsp://192.168.1.1:554/stream
```

### 6. 모델 평가

학습 완료 후 정량적 평가와 정성적 평가를 순서대로 실행합니다.

#### 정량적 평가 (숫자 지표)
```bash
python src/evaluation/evaluate.py --model models/highway_detector_best.pt

# 빠른 평가 (임계값 분석 생략)
python src/evaluation/evaluate.py --model models/highway_detector_best.pt --skip_threshold
```

**출력 결과 (`runs/eval/`):**
| 파일 | 내용 |
|------|------|
| `eval_summary_*.txt` | 전체 요약 리포트 (콘솔 출력과 동일) |
| `eval_*.json` | 모든 지표 (프로그래밍 활용용) |
| `eval_per_class_*.csv` | 클래스별 지표 스프레드시트 |

**측정 지표:**

| 지표 | 설명 | 목표값 |
|------|------|--------|
| **mAP@0.5** | IoU 0.5 기준 평균 정밀도 | > 0.70 |
| **mAP@0.5:0.95** | 다양한 IoU 기준 평균 (엄격) | > 0.50 |
| **Precision** | 감지한 것 중 실제 위험 비율 | 오탐 감소 |
| **Recall** | 실제 위험 중 감지한 비율 | **CRITICAL 클래스는 최우선** |
| **F1-Score** | Precision과 Recall의 조화 평균 | > 0.75 |
| **FPS** | 초당 처리 프레임 수 | > 15 (실시간 기준) |

> **핵심 전략:** `accident`, `fire`는 미탐(놓치는 것)이 오탐보다 훨씬 위험하므로 **Recall을 우선** 확인합니다.
> `stopped_vehicle`은 잦은 오탐이 운영자 피로를 유발하므로 **Precision을 우선** 확인합니다.

**임계값 민감도 분석** — conf 0.25 ~ 0.75 범위에서 자동 측정하여 최적 임계값 선정에 활용합니다.

```
conf=0.25  mAP50=0.7821  P=0.7234  R=0.8901  F1=0.7981   ← Recall 높음 (오탐 증가)
conf=0.45  mAP50=0.7654  P=0.8012  R=0.8234  F1=0.8121   ← 기본값 (균형)
conf=0.65  mAP50=0.7102  P=0.8891  R=0.7012  F1=0.7843   ← Precision 높음 (미탐 증가)
```

---

#### 정성적 평가 (시각적 분석)
```bash
python src/evaluation/visualize_results.py --model models/highway_detector_best.pt

# 샘플 수 지정
python src/evaluation/visualize_results.py --model models/highway_detector_best.pt --n_samples 50
```

**출력 결과 (`runs/eval/qualitative_날짜/`):**

| 파일/폴더 | 내용 | 확인 포인트 |
|-----------|------|-------------|
| `tp/` | 정확히 감지된 케이스 (초록 박스) | 잘 잡히는 패턴 파악 |
| `fp/` | 오탐 케이스 (빨간 박스 + GT 초록) | 터널 조명·안개 오탐 여부 |
| `fn/` | 미탐 케이스 (파란 박스) | 원거리·야간·역광 실패 분석 |
| `confusion_matrix.png` | 클래스 간 혼동 행렬 | fire↔smoke 혼동 빈도 |
| `pr_curve.png` | 클래스별 Precision-Recall 포인트 | 클래스별 성능 한눈에 비교 |
| `detection_grid.jpg` | GT vs Prediction 4×4 그리드 | 전반적인 감지 품질 확인 |

**실패 케이스 분석 체크리스트:**
```
FP (오탐) 주요 원인:
  □ 터널 조명 → fire 오탐
  □ 안개/박무 → smoke 오탐
  □ 공사 차량/크레인 → accident 오탐

FN (미탐) 주요 원인:
  □ 원거리(카메라에서 멀리 떨어진) 화재
  □ 야간 사고 (조명 부족)
  □ 부분 가림(occlusion) - 다른 차량에 가려진 사고
  □ 역광 상황
```

---

### 7. 팝업 경고 테스트
```bash
python src/alert/popup_alert.py
```

## 프로젝트 구조
```
VisionMachine_Test/
├── data/
│   ├── raw/              # 원본 CCTV 영상
│   ├── images/           # 추출된 프레임
│   ├── labels/           # YOLO 라벨 (.txt)
│   └── datasets/         # train/val/test 분리 완료
├── models/               # 학습된 모델 (.pt)
├── src/
│   ├── data_prep/        # 데이터 전처리
│   ├── training/         # 학습 스크립트
│   ├── inference/        # 추론 & 실시간 감지
│   ├── evaluation/       # 평가 스크립트
│   │   ├── evaluate.py          # 정량적 평가
│   │   └── visualize_results.py # 정성적 평가 시각화
│   └── alert/            # 팝업 경고 시스템
├── configs/
│   ├── dataset.yaml      # 데이터셋 설정
│   └── train_config.yaml # 학습 하이퍼파라미터
├── runs/
│   ├── train/            # 학습 결과 (자동 생성)
│   └── eval/             # 평가 결과 (자동 생성)
└── requirements.txt
```

## 추천 공개 데이터셋
- **D-Fire** (화재/연기): https://github.com/gaiasd/DFireDataset
- **Roboflow - Fire Detection**: https://universe.roboflow.com/nuri-gjsug/fire-and-smoke-detection-bkxl5
- **Roboflow - Accident Detection**: https://universe.roboflow.com/accident-detection-ffdkg/accident-detection-8dvh5
