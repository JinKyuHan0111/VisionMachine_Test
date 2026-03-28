"""
YOLOv8 추론 엔진
학습된 모델로 이미지/영상에서 위험 상황 감지
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 클래스별 표시 색상 (BGR)
CLASS_COLORS = {
    "fire":            (0,   69,  255),  # 주황
    "smoke":           (128, 128, 128),  # 회색
}

DEFAULT_CONF = 0.25  # 기본 신뢰도 임계값 (Recall 우선)


class HighwayDetector:
    """고속도로 위험 상황 감지기"""

    def __init__(self, model_path: str, conf: float = DEFAULT_CONF,
                 device: str = "0"):
        """
        Args:
            model_path: 학습된 모델 경로 (.pt)
            conf: 신뢰도 임계값
            device: 추론 디바이스 ('0'=GPU, 'cpu'=CPU)
        """
        self.model_path = Path(model_path)
        self.conf = conf
        self.device = device
        self.model = None
        self.class_names = []

        self._load_model()

    def _load_model(self):
        """모델 로드"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names
        print(f"모델 로드 완료: {self.model_path.name}")
        print(f"  클래스: {list(self.class_names.values())}")

    def detect_image(self, image: np.ndarray) -> list:
        """
        단일 이미지 추론

        Returns:
            list of dict: [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]
        """
        results = self.model(image, conf=self.conf, device=self.device,
                             verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "class": self.class_names[cls_id],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                })

        return detections

    def draw_detections(self, image: np.ndarray, detections: list,
                        show_conf: bool = True) -> np.ndarray:
        """감지 결과를 이미지에 시각화"""
        vis = image.copy()

        for det in detections:
            cls = det["class"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]
            color = CLASS_COLORS.get(cls, (0, 255, 0))

            # 바운딩 박스
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 라벨 배경
            label = f"{cls} {conf:.2f}" if show_conf else cls
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

            # 라벨 텍스트
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)

        return vis

    def is_dangerous(self, detections: list) -> tuple[bool, list]:
        """
        위험 상황 여부 판별

        Returns:
            (is_dangerous, dangerous_detections)
        """
        dangerous_classes = {"fire", "smoke"}
        dangerous = [d for d in detections if d["class"] in dangerous_classes]
        return len(dangerous) > 0, dangerous
