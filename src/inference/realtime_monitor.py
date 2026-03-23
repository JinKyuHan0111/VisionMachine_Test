"""
실시간 고속도로 CCTV 모니터링 시스템
웹캠, 동영상 파일, RTSP 스트림 지원
"""

import cv2
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.detector import HighwayDetector
from src.alert.popup_alert import trigger_alert

# 기본 모델 경로
DEFAULT_MODEL = PROJECT_ROOT / "models" / "highway_detector_best.pt"

# 감지 트리거 임계값 (연속 N프레임 감지 시 경고)
TRIGGER_FRAMES = 3


class CCTVMonitor:
    """실시간 CCTV 모니터링"""

    def __init__(self, source, model_path: str,
                 camera_name: str = "카메라 1",
                 conf: float = 0.45,
                 save_clips: bool = False):
        """
        Args:
            source: 영상 소스 (0=웹캠, 파일경로, 'rtsp://...')
            model_path: 모델 경로
            camera_name: 카메라 식별 이름
            conf: 신뢰도 임계값
            save_clips: 위험 상황 클립 저장 여부
        """
        self.source = source
        self.camera_name = camera_name
        self.save_clips = save_clips
        self.detector = HighwayDetector(model_path, conf=conf)

        # 연속 감지 카운터 {class_name: count}
        self._consecutive: dict = {}
        self._is_running = False

    def run(self):
        """모니터링 시작"""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[오류] 영상 소스를 열 수 없습니다: {self.source}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        print(f"\n[모니터링 시작]")
        print(f"  소스: {self.source}")
        print(f"  해상도: {width}x{height} @ {fps:.0f}fps")
        print(f"  카메라: {self.camera_name}")
        print(f"  종료: 'q' 키\n")

        self._is_running = True
        frame_count = 0
        fps_timer = time.time()
        display_fps = 0.0

        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                print("영상 스트림 종료")
                break

            frame_count += 1

            # 매 프레임 추론 (GPU이므로 충분히 빠름)
            detections = self.detector.detect_image(frame)
            is_danger, danger_dets = self.detector.is_dangerous(detections)

            # 위험 감지 처리
            if is_danger:
                self._handle_danger(danger_dets)
            else:
                self._consecutive.clear()

            # 시각화
            vis_frame = self.detector.draw_detections(frame, detections)
            vis_frame = self._draw_overlay(vis_frame, display_fps, is_danger)

            # FPS 계산 (1초마다 업데이트)
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_timer
                display_fps = 30.0 / elapsed if elapsed > 0 else 0
                fps_timer = time.time()

            # 화면 표시
            cv2.imshow(f"Highway CCTV Monitor - {self.camera_name}", vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("모니터링 종료")
                break

        cap.release()
        cv2.destroyAllWindows()

    def _handle_danger(self, detections: list):
        """위험 상황 처리 - 연속 감지 시 경고 발송"""
        for det in detections:
            cls = det["class"]
            self._consecutive[cls] = self._consecutive.get(cls, 0) + 1

            if self._consecutive[cls] == TRIGGER_FRAMES:
                print(f"[경고] {cls} 감지! 신뢰도: {det['confidence']:.1%} "
                      f"위치: {self.camera_name} "
                      f"({datetime.now().strftime('%H:%M:%S')})")
                trigger_alert(
                    class_name=cls,
                    confidence=det["confidence"],
                    location=self.camera_name,
                )

    def _draw_overlay(self, frame, fps: float, is_danger: bool):
        """HUD 오버레이 그리기"""
        h, w = frame.shape[:2]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 상단 정보 바
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"{self.camera_name}  |  FPS: {fps:.1f}  |  {timestamp}",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # 위험 상태 표시
        if is_danger:
            status_color = (0, 0, 255)
            status_text = "DANGER DETECTED"
        else:
            status_color = (0, 200, 0)
            status_text = "NORMAL"

        cv2.putText(frame, status_text, (w - 200, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2,
                    cv2.LINE_AA)

        # 'q' 종료 안내
        cv2.putText(frame, "Press 'q' to quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame


def main():
    parser = argparse.ArgumentParser(description="고속도로 CCTV 실시간 모니터링")
    parser.add_argument("--source", type=str, default="0",
                        help="영상 소스 (0=웹캠, 파일경로, rtsp://...)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help="모델 경로 (.pt)")
    parser.add_argument("--camera", type=str, default="카메라 1",
                        help="카메라 이름")
    parser.add_argument("--conf", type=float, default=0.45,
                        help="신뢰도 임계값 (0.0~1.0)")
    args = parser.parse_args()

    # 숫자면 int로 변환 (웹캠 인덱스)
    source = int(args.source) if args.source.isdigit() else args.source

    monitor = CCTVMonitor(
        source=source,
        model_path=args.model,
        camera_name=args.camera,
        conf=args.conf,
    )
    monitor.run()


if __name__ == "__main__":
    main()
