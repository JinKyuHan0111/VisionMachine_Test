"""
정량적 평가 스크립트
mAP, Precision, Recall, F1-Score, FPS 측정 후 리포트 저장
"""

import time
import json
import csv
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = ["accident", "fire", "smoke", "stopped_vehicle", "debris"]

# 클래스별 평가 우선순위 (위험도 기반)
CLASS_PRIORITY = {
    "accident":        "CRITICAL",
    "fire":            "CRITICAL",
    "smoke":           "WARNING",
    "debris":          "WARNING",
    "stopped_vehicle": "INFO",
}


def measure_fps(model: YOLO, img_size: int = 640, n_runs: int = 100,
                device: str = "0") -> dict:
    """
    추론 속도 측정 (GPU 워밍업 포함)

    Returns:
        {"fps": float, "ms_per_frame": float}
    """
    dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    # GPU 워밍업 (첫 실행은 느림)
    for _ in range(10):
        model(dummy, device=device, verbose=False)

    # 실제 측정
    start = time.perf_counter()
    for _ in range(n_runs):
        model(dummy, device=device, verbose=False)
    elapsed = time.perf_counter() - start

    ms = (elapsed / n_runs) * 1000
    fps = 1000 / ms
    return {"fps": round(fps, 1), "ms_per_frame": round(ms, 2)}


def run_validation(model: YOLO, data_yaml: str, img_size: int = 640,
                   conf: float = 0.45, iou: float = 0.5,
                   device: str = "0") -> dict:
    """
    YOLOv8 공식 val() 실행 → 클래스별 지표 반환

    Returns:
        dict: 클래스별 및 전체 mAP, Precision, Recall
    """
    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
        plots=False,
    )

    metrics = {}

    # 전체 지표
    metrics["overall"] = {
        "mAP50":    round(float(results.box.map50), 4),
        "mAP50_95": round(float(results.box.map), 4),
        "precision": round(float(results.box.mp), 4),
        "recall":    round(float(results.box.mr), 4),
        "f1": round(
            2 * float(results.box.mp) * float(results.box.mr)
            / (float(results.box.mp) + float(results.box.mr) + 1e-8), 4
        ),
    }

    # 클래스별 지표
    metrics["per_class"] = {}
    ap50_per_class = results.box.ap50   # shape: (n_classes,)
    p_per_class    = results.box.p      # shape: (n_classes,)
    r_per_class    = results.box.r      # shape: (n_classes,)

    names = results.names  # {0: 'accident', 1: 'fire', ...}
    for idx, name in names.items():
        if idx >= len(ap50_per_class):
            continue
        p = float(p_per_class[idx])
        r = float(r_per_class[idx])
        f1 = 2 * p * r / (p + r + 1e-8)
        metrics["per_class"][name] = {
            "AP50":      round(float(ap50_per_class[idx]), 4),
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
            "priority":  CLASS_PRIORITY.get(name, "INFO"),
        }

    return metrics


def evaluate_threshold_sensitivity(model: YOLO, data_yaml: str,
                                   img_size: int = 640,
                                   device: str = "0") -> list:
    """
    신뢰도 임계값별 Precision/Recall 변화 측정
    → 최적 임계값 선정에 활용
    """
    thresholds = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
    rows = []

    print("  임계값 민감도 분석 중...")
    for conf in thresholds:
        res = model.val(
            data=data_yaml, imgsz=img_size, conf=conf,
            iou=0.5, device=device, verbose=False, plots=False,
        )
        p = float(res.box.mp)
        r = float(res.box.mr)
        f1 = 2 * p * r / (p + r + 1e-8)
        rows.append({
            "conf_threshold": conf,
            "mAP50":     round(float(res.box.map50), 4),
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
        })
        print(f"    conf={conf:.2f}  mAP50={rows[-1]['mAP50']:.4f}  "
              f"P={rows[-1]['precision']:.4f}  R={rows[-1]['recall']:.4f}  "
              f"F1={rows[-1]['f1']:.4f}")

    return rows


def save_report(metrics: dict, fps_info: dict, threshold_rows: list,
                save_dir: Path, model_name: str):
    """평가 결과를 JSON + CSV + TXT 로 저장"""
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── JSON ────────────────────────────────────────────
    report = {
        "model": model_name,
        "timestamp": timestamp,
        "fps": fps_info,
        "metrics": metrics,
        "threshold_sensitivity": threshold_rows,
    }
    json_path = save_dir / f"eval_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── CSV (클래스별) ────────────────────────────────
    csv_path = save_dir / f"eval_per_class_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class", "priority", "AP50", "precision",
                           "recall", "f1"]
        )
        writer.writeheader()
        for cls, v in metrics["per_class"].items():
            writer.writerow({"class": cls, **v})

    # ── TXT (사람이 읽기 좋은 요약) ──────────────────
    txt_path = save_dir / f"eval_summary_{timestamp}.txt"
    overall = metrics["overall"]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write("  Highway CCTV Detector - 평가 리포트\n")
        f.write("=" * 55 + "\n")
        f.write(f"  모델    : {model_name}\n")
        f.write(f"  일시    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 55 + "\n")
        f.write("[전체 성능]\n")
        f.write(f"  mAP@0.5      : {overall['mAP50']:.4f}\n")
        f.write(f"  mAP@0.5:0.95 : {overall['mAP50_95']:.4f}\n")
        f.write(f"  Precision    : {overall['precision']:.4f}\n")
        f.write(f"  Recall       : {overall['recall']:.4f}\n")
        f.write(f"  F1-Score     : {overall['f1']:.4f}\n")
        f.write(f"  FPS          : {fps_info['fps']} ({fps_info['ms_per_frame']}ms/frame)\n")
        f.write("-" * 55 + "\n")
        f.write("[클래스별 성능]\n")
        f.write(f"  {'클래스':<18} {'위험도':<10} {'AP50':>6} {'P':>6} {'R':>6} {'F1':>6}\n")
        f.write("  " + "-" * 50 + "\n")
        for cls, v in metrics["per_class"].items():
            f.write(f"  {cls:<18} {v['priority']:<10} "
                    f"{v['AP50']:>6.4f} {v['precision']:>6.4f} "
                    f"{v['recall']:>6.4f} {v['f1']:>6.4f}\n")
        f.write("-" * 55 + "\n")
        f.write("[임계값별 성능 (conf threshold sensitivity)]\n")
        f.write(f"  {'conf':>6} {'mAP50':>7} {'P':>7} {'R':>7} {'F1':>7}\n")
        f.write("  " + "-" * 38 + "\n")
        for row in threshold_rows:
            f.write(f"  {row['conf_threshold']:>6.2f} "
                    f"{row['mAP50']:>7.4f} "
                    f"{row['precision']:>7.4f} "
                    f"{row['recall']:>7.4f} "
                    f"{row['f1']:>7.4f}\n")
        f.write("=" * 55 + "\n")

    print(f"\n[리포트 저장 완료]")
    print(f"  JSON : {json_path}")
    print(f"  CSV  : {csv_path}")
    print(f"  TXT  : {txt_path}")

    return txt_path


def print_summary(metrics: dict, fps_info: dict):
    """콘솔 요약 출력"""
    overall = metrics["overall"]
    print("\n" + "=" * 55)
    print("  평가 결과 요약")
    print("=" * 55)
    print(f"  mAP@0.5      : {overall['mAP50']:.4f}  "
          f"({'Good' if overall['mAP50'] > 0.7 else 'Need improvement'})")
    print(f"  mAP@0.5:0.95 : {overall['mAP50_95']:.4f}")
    print(f"  Precision    : {overall['precision']:.4f}")
    print(f"  Recall       : {overall['recall']:.4f}  "
          f"({'Good' if overall['recall'] > 0.75 else 'Low - 미탐 주의'})")
    print(f"  F1-Score     : {overall['f1']:.4f}")
    print(f"  FPS          : {fps_info['fps']}  "
          f"({'실시간 가능' if fps_info['fps'] >= 15 else '실시간 불가'})")
    print("-" * 55)

    # CRITICAL 클래스 강조
    print("  [CRITICAL 클래스 Recall]")
    for cls, v in metrics["per_class"].items():
        if v["priority"] == "CRITICAL":
            status = "OK" if v["recall"] > 0.75 else "LOW - 미탐 위험"
            print(f"    {cls:<18}: {v['recall']:.4f}  ({status})")
    print("=" * 55)


def evaluate(model_path: str, data_yaml: str = None,
             conf: float = 0.45, device: str = "0",
             skip_threshold: bool = False):
    """
    전체 평가 실행

    Args:
        model_path: 모델 경로 (.pt)
        data_yaml: dataset.yaml 경로
        conf: 기본 신뢰도 임계값
        device: 디바이스
        skip_threshold: 임계값 민감도 분석 건너뛰기 (빠른 평가)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[오류] 모델 없음: {model_path}")
        return

    if data_yaml is None:
        data_yaml = str(PROJECT_ROOT / "configs" / "dataset.yaml")

    print(f"[평가 시작]")
    print(f"  모델  : {model_path.name}")
    print(f"  데이터: {data_yaml}")
    print(f"  장치  : GPU {device}\n")

    model = YOLO(str(model_path))

    # 1. FPS 측정
    print("[1/3] FPS 측정 중...")
    fps_info = measure_fps(model, device=device)
    print(f"  → {fps_info['fps']} FPS ({fps_info['ms_per_frame']}ms/frame)")

    # 2. 정량 지표
    print("\n[2/3] 정량 지표 계산 중 (mAP / Precision / Recall)...")
    metrics = run_validation(model, data_yaml, conf=conf, device=device)

    # 3. 임계값 민감도
    threshold_rows = []
    if not skip_threshold:
        print("\n[3/3] 임계값 민감도 분석 중...")
        threshold_rows = evaluate_threshold_sensitivity(
            model, data_yaml, device=device
        )
    else:
        print("\n[3/3] 임계값 분석 건너뜀 (--skip_threshold)")

    # 결과 출력 & 저장
    print_summary(metrics, fps_info)

    save_dir = PROJECT_ROOT / "runs" / "eval"
    txt_path = save_report(metrics, fps_info, threshold_rows,
                           save_dir, model_path.name)

    with open(txt_path, "r", encoding="utf-8") as f:
        print("\n" + f.read())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="정량적 평가")
    parser.add_argument("--model", type=str,
                        default=str(PROJECT_ROOT / "models" / "highway_detector_best.pt"))
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--skip_threshold", action="store_true",
                        help="임계값 민감도 분석 생략 (빠른 평가)")
    args = parser.parse_args()

    evaluate(args.model, args.data, args.conf, args.device, args.skip_threshold)
