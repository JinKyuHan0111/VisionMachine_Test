"""
정성적 평가 시각화 스크립트
- TP / FP / FN 케이스 이미지 추출
- 혼동 행렬 (Confusion Matrix)
- PR Curve (Precision-Recall Curve)
- 클래스별 감지 샘플 그리드
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서도 동작
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 클래스별 색상 (BGR for OpenCV / RGB for matplotlib)
CLASS_COLORS_BGR = {
    "accident":        (0,   0,   220),
    "fire":            (0,   100, 255),
    "smoke":           (140, 140, 140),
    "stopped_vehicle": (220, 130,  20),
    "debris":          (0,   180, 255),
}
CLASS_COLORS_RGB = {k: (v[2], v[1], v[0]) for k, v in CLASS_COLORS_BGR.items()}

IOU_THRESHOLD = 0.5


def box_iou(box1: list, box2: list) -> float:
    """두 바운딩 박스의 IoU 계산 (xyxy 형식)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(label_path: Path, img_w: int, img_h: int,
                      class_names: dict) -> list:
    """YOLO 형식 라벨 파일 로드 → xyxy 절대좌표로 변환"""
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            boxes.append({
                "class": class_names.get(cls_id, str(cls_id)),
                "bbox": [x1, y1, x2, y2],
            })
    return boxes


def classify_detections(preds: list, gts: list) -> tuple:
    """
    예측 결과를 TP / FP / FN 으로 분류

    Returns:
        (tp_list, fp_list, fn_list)
        각 항목: {"class": str, "bbox": [...], "confidence"?: float}
    """
    matched_gt = set()
    tp, fp = [], []

    for pred in preds:
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(gts):
            if gt["class"] != pred["class"] or i in matched_gt:
                continue
            iou = box_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= IOU_THRESHOLD:
            tp.append(pred)
            matched_gt.add(best_idx)
        else:
            fp.append(pred)

    fn = [gt for i, gt in enumerate(gts) if i not in matched_gt]
    return tp, fp, fn


def draw_boxes(image: np.ndarray, boxes: list, color: tuple,
               label_prefix: str = "", thickness: int = 2) -> np.ndarray:
    """바운딩 박스 그리기"""
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        cls = box.get("class", "")
        conf = box.get("confidence", None)
        label = f"{label_prefix}{cls}"
        if conf is not None:
            label += f" {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return image


def extract_tp_fp_fn_samples(model: YOLO, images_dir: Path,
                              labels_dir: Path, save_dir: Path,
                              n_samples: int = 30, conf: float = 0.45,
                              device: str = "0"):
    """
    TP/FP/FN 샘플 이미지 추출 & 저장

    저장 구조:
      save_dir/
        tp/  - 정확히 감지된 케이스
        fp/  - 오탐 케이스
        fn/  - 미탐 케이스
    """
    for folder in ["tp", "fp", "fn"]:
        (save_dir / folder).mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png"}
    images = sorted([f for f in images_dir.iterdir()
                     if f.suffix.lower() in img_exts])[:n_samples * 3]

    counts = {"tp": 0, "fp": 0, "fn": 0}
    class_names = model.names

    print(f"  {len(images)}개 이미지 분석 중...")
    for img_path in images:
        if all(v >= n_samples for v in counts.values()):
            break

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        # 추론
        results = model(image, conf=conf, device=device, verbose=False)
        preds = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                preds.append({
                    "class": class_names[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox": list(map(int, box.xyxy[0])),
                })

        # GT 로드
        label_path = labels_dir / (img_path.stem + ".txt")
        gts = load_ground_truth(label_path, w, h, class_names)

        if not preds and not gts:
            continue

        tp, fp, fn = classify_detections(preds, gts)

        def save_sample(case_type: str, vis: np.ndarray):
            if counts[case_type] < n_samples:
                out = save_dir / case_type / f"{img_path.stem}_{case_type}.jpg"
                cv2.imwrite(str(out), vis)
                counts[case_type] += 1

        # TP 저장
        if tp and counts["tp"] < n_samples:
            vis = image.copy()
            draw_boxes(vis, tp, (0, 200, 0), "TP: ")
            save_sample("tp", vis)

        # FP 저장
        if fp and counts["fp"] < n_samples:
            vis = image.copy()
            draw_boxes(vis, fp, (0, 0, 255), "FP: ")
            draw_boxes(vis, gts, (0, 200, 0), "GT: ", thickness=1)
            save_sample("fp", vis)

        # FN 저장
        if fn and counts["fn"] < n_samples:
            vis = image.copy()
            draw_boxes(vis, fn, (255, 0, 0), "FN: ")
            draw_boxes(vis, tp, (0, 200, 0), "TP: ", thickness=1)
            save_sample("fn", vis)

    print(f"  TP: {counts['tp']}장, FP: {counts['fp']}장, FN: {counts['fn']}장")
    return counts


def plot_confusion_matrix(model: YOLO, data_yaml: str, save_dir: Path,
                          conf: float = 0.45, device: str = "0"):
    """YOLOv8 내장 혼동 행렬 생성 & 저장"""
    print("  혼동 행렬 생성 중...")
    results = model.val(
        data=data_yaml,
        conf=conf,
        device=device,
        plots=True,
        save_dir=str(save_dir),
        verbose=False,
    )
    # YOLOv8 val()은 confusion_matrix.png 를 save_dir에 자동 저장
    src = Path(results.save_dir) / "confusion_matrix.png"
    dest = save_dir / "confusion_matrix.png"
    if src.exists() and src != dest:
        import shutil
        shutil.copy2(src, dest)
        print(f"  저장: {dest}")
    return dest


def plot_pr_curve(model: YOLO, data_yaml: str, save_dir: Path,
                  device: str = "0"):
    """
    Precision-Recall Curve 생성
    YOLOv8 val()로 클래스별 AP 계산 후 matplotlib으로 시각화
    """
    print("  PR Curve 생성 중...")
    results = model.val(
        data=data_yaml, device=device, verbose=False, plots=False
    )

    class_names = results.names
    ap50 = results.box.ap50        # (n_classes,)
    precision = results.box.p      # (n_classes,)
    recall = results.box.r         # (n_classes,)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.get_cmap("tab10")

    for idx, name in class_names.items():
        if idx >= len(ap50):
            continue
        p = float(precision[idx])
        r = float(recall[idx])
        ap = float(ap50[idx])
        color = cmap(idx % 10)
        ax.scatter(r, p, color=color, s=120, zorder=5)
        ax.annotate(
            f"{name}\nAP={ap:.3f}",
            (r, p),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
            color=color,
        )

    # 전체 평균
    overall_p = float(results.box.mp)
    overall_r = float(results.box.mr)
    ax.scatter(overall_r, overall_p, marker="*", s=250,
               color="black", zorder=6, label=f"Overall (mAP50={results.box.map50:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall (per class @ conf=0.45)", fontsize=13)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "pr_curve.png"
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  저장: {out}")
    return out


def plot_detection_grid(images_dir: Path, labels_dir: Path,
                        model: YOLO, save_dir: Path,
                        n_rows: int = 4, n_cols: int = 4,
                        conf: float = 0.45, device: str = "0"):
    """
    감지 결과 그리드 이미지 생성 (GT vs Prediction 비교)
    """
    print("  감지 샘플 그리드 생성 중...")
    img_exts = {".jpg", ".jpeg", ".png"}
    images = sorted([f for f in images_dir.iterdir()
                     if f.suffix.lower() in img_exts])
    images = images[:n_rows * n_cols]

    if not images:
        print("  [건너뜀] 이미지 없음")
        return

    class_names = model.names
    cell_size = (320, 240)  # 각 셀 크기
    grid_w = n_cols * cell_size[0]
    grid_h = n_rows * cell_size[1]
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i, img_path in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        # GT
        label_path = labels_dir / (img_path.stem + ".txt")
        gts = load_ground_truth(label_path, w, h, class_names)
        draw_boxes(image, gts, (0, 200, 0), "GT:")

        # Prediction
        results = model(image, conf=conf, device=device, verbose=False)
        preds = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                preds.append({
                    "class": class_names[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox": list(map(int, box.xyxy[0])),
                })
        draw_boxes(image, preds, (0, 0, 220), "P:")

        # 셀에 배치
        cell = cv2.resize(image, cell_size)
        y_start = row * cell_size[1]
        x_start = col * cell_size[0]
        grid[y_start:y_start + cell_size[1],
             x_start:x_start + cell_size[0]] = cell

    # 범례 추가
    legend_h = 30
    legend = np.zeros((legend_h, grid_w, 3), dtype=np.uint8)
    cv2.putText(legend,
                "Green=Ground Truth  |  Red=Prediction",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)
    grid = np.vstack([grid, legend])

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "detection_grid.jpg"
    cv2.imwrite(str(out), grid)
    print(f"  저장: {out}")
    return out


def run_qualitative_eval(model_path: str, images_dir: str = None,
                          labels_dir: str = None, data_yaml: str = None,
                          conf: float = 0.45, device: str = "0",
                          n_samples: int = 30):
    """정성적 평가 전체 실행"""
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[오류] 모델 없음: {model_path}")
        return

    if data_yaml is None:
        data_yaml = str(PROJECT_ROOT / "configs" / "dataset.yaml")
    if images_dir is None:
        images_dir = PROJECT_ROOT / "data" / "datasets" / "images" / "test"
    if labels_dir is None:
        labels_dir = PROJECT_ROOT / "data" / "datasets" / "labels" / "test"

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = PROJECT_ROOT / "runs" / "eval" / f"qualitative_{timestamp}"

    print(f"[정성적 평가 시작]")
    print(f"  모델  : {model_path.name}")
    print(f"  이미지: {images_dir}")
    print()

    model = YOLO(str(model_path))

    # 1. TP/FP/FN 샘플
    print("[1/4] TP/FP/FN 샘플 추출...")
    extract_tp_fp_fn_samples(
        model, images_dir, labels_dir,
        save_dir / "tp_fp_fn", n_samples, conf, device
    )

    # 2. 혼동 행렬
    print("\n[2/4] 혼동 행렬...")
    plot_confusion_matrix(model, data_yaml, save_dir, conf, device)

    # 3. PR Curve
    print("\n[3/4] PR Curve...")
    plot_pr_curve(model, data_yaml, save_dir, device)

    # 4. 감지 그리드
    print("\n[4/4] 감지 샘플 그리드...")
    plot_detection_grid(images_dir, labels_dir, model, save_dir,
                        conf=conf, device=device)

    print(f"\n[완료] 모든 시각화 결과: {save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="정성적 평가 시각화")
    parser.add_argument("--model", type=str,
                        default=str(PROJECT_ROOT / "models" / "highway_detector_best.pt"))
    parser.add_argument("--images", type=str, default=None,
                        help="테스트 이미지 폴더")
    parser.add_argument("--labels", type=str, default=None,
                        help="테스트 라벨 폴더")
    parser.add_argument("--data", type=str, default=None,
                        help="dataset.yaml 경로")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--n_samples", type=int, default=30,
                        help="TP/FP/FN 샘플 최대 개수")
    args = parser.parse_args()

    run_qualitative_eval(
        model_path=args.model,
        images_dir=args.images,
        labels_dir=args.labels,
        data_yaml=args.data,
        conf=args.conf,
        device=args.device,
        n_samples=args.n_samples,
    )
