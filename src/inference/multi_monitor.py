"""
다중 CCTV 실시간 모니터링
모델을 한 번만 로드하고 여러 카메라에서 공유
imshow는 메인 스레드에서만 처리 (Windows 호환)
--headless (또는 HEADLESS=1): 화면/팝업 없이 감지 + DB/캡처 저장만 (Docker, 서버용)
"""

import cv2
import time
import threading
import queue
import argparse
import os
import signal
import sys
import re
import math
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.detector import HighwayDetector
from src.alert.popup_alert import trigger_alert

DEFAULT_MODEL   = PROJECT_ROOT / "models" / "fire_detector_best.pt"
TRIGGER_FRAMES  = 3
MAX_RECONNECT   = 10
RECONNECT_DELAY = 5
MAX_FAIL_FRAMES = 30


class CameraWorker(threading.Thread):
    """단일 카메라 캡처 + 추론 스레드 (화면 표시는 메인 스레드에서)"""

    def __init__(self, source, detector: HighwayDetector,
                 camera_name: str, frame_skip: int = 1, headless: bool = False):
        super().__init__(daemon=True)
        self.source      = source
        self.detector    = detector
        self.camera_name = camera_name
        self.frame_skip  = frame_skip
        self.headless    = headless

        self._is_running    = False
        self._current_frame = None
        self._consecutive: dict = {}
        self._infer_lock = threading.Lock()

        # 메인 스레드로 프레임 전달하는 큐 (최신 1프레임만 유지)
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)

        self._capture_dir = PROJECT_ROOT / "data" / "captures"
        self._capture_dir.mkdir(parents=True, exist_ok=True)

        self._display_fps = 0.0

    def stop(self):
        self._is_running = False

    def _open_stream(self):
        attempt = 0
        while self._is_running:
            attempt += 1
            source_str = str(self.source)
            if isinstance(self.source, str) and not any(
                source_str.startswith(p)
                for p in ("rtsp://", "rtsps://", "http://", "https://", "rtmp://")
            ) and not Path(source_str).exists():
                print(f"[보안] 허용되지 않는 소스 차단: {self.source}")
                return None

            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                print(f"[{self.camera_name}] 연결 성공 {w}x{h} @ {fps:.0f}fps")
                return cap

            cap.release()
            if attempt >= MAX_RECONNECT:
                print(f"[{self.camera_name}] 최대 재연결 횟수 초과")
                return None
            print(f"[{self.camera_name}] {RECONNECT_DELAY}초 후 재연결...")
            time.sleep(RECONNECT_DELAY)
        return None

    def _save_capture(self, cls: str, detections: list):
        if self._current_frame is None:
            return None
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        cam = re.sub(r"[^\w\-]", "_", self.camera_name)
        save_path = self._capture_dir / f"{ts}_{cam}_{cls}.jpg"
        if not save_path.resolve().is_relative_to(self._capture_dir.resolve()):
            return None
        vis = self.detector.draw_detections(self._current_frame, detections)
        cv2.imwrite(str(save_path), vis)
        return str(save_path)

    def _handle_danger(self, detections: list):
        for det in detections:
            cls = det["class"]
            self._consecutive[cls] = self._consecutive.get(cls, 0) + 1
            if self._consecutive[cls] == TRIGGER_FRAMES:
                frame_path = self._save_capture(cls, detections)
                print(f"[경고] {cls} | {self.camera_name} | "
                      f"{det['confidence']:.1%} | {datetime.now().strftime('%H:%M:%S')}")
                trigger_alert(class_name=cls, confidence=det["confidence"],
                              location=self.camera_name, frame_path=frame_path)

    def _draw_overlay(self, frame):
        h, w = frame.shape[:2]
        ts   = datetime.now().strftime("%H:%M:%S")
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"{self.camera_name}  FPS:{self._display_fps:.1f}  {ts}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def run(self):
        self._is_running = True
        print(f"[{self.camera_name}] 모니터링 시작 (frame_skip={self.frame_skip})")

        while self._is_running:
            cap = self._open_stream()
            if cap is None:
                break

            frame_count = 0
            fail_count  = 0
            fps_timer   = time.time()

            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    fail_count += 1
                    if fail_count >= MAX_FAIL_FRAMES:
                        print(f"[{self.camera_name}] 스트림 끊김 - 재연결 시도")
                        break
                    time.sleep(0.03)
                    continue

                fail_count  = 0
                frame_count += 1
                self._current_frame = frame

                # frame_skip마다 추론
                if frame_count % self.frame_skip == 0:
                    with self._infer_lock:
                        detections = self.detector.detect_image(frame)
                    is_danger, danger_dets = self.detector.is_dangerous(detections)

                    if is_danger:
                        self._handle_danger(danger_dets)
                    else:
                        self._consecutive.clear()

                    if self.headless:
                        continue  # 화면이 없으면 시각화/프레임 전달 생략 (CPU 절약)
                    vis = self.detector.draw_detections(frame, detections)
                else:
                    if self.headless:
                        continue
                    vis = frame.copy()

                # FPS 계산
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_timer
                    self._display_fps = 30.0 / elapsed if elapsed > 0 else 0
                    fps_timer = time.time()

                vis = self._draw_overlay(vis)

                # 메인 스레드로 전달 (가득 차면 이전 프레임 버림)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(vis)

            cap.release()

        print(f"[{self.camera_name}] 종료")


def make_grid(frames: list, names: list, cell_w: int = 640, cell_h: int = 360) -> np.ndarray:
    """여러 프레임을 그리드로 합치기"""
    n    = len(frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    cells = []
    for i, (frame, name) in enumerate(zip(frames, names)):
        cell = cv2.resize(frame, (cell_w, cell_h))
        # 카메라 이름 라벨
        cv2.rectangle(cell, (0, 0), (cell_w, 28), (0, 0, 0), -1)
        cv2.putText(cell, name, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(cell)

    # 빈 셀 채우기
    blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    while len(cells) < rows * cols:
        cells.append(blank)

    rows_imgs = []
    for r in range(rows):
        row_cells = cells[r * cols: (r + 1) * cols]
        rows_imgs.append(np.hstack(row_cells))

    return np.vstack(rows_imgs)


def _run_headless(workers: list):
    """화면 없이 워커가 끝날 때까지 대기. docker stop(SIGTERM)에도 정상 종료"""
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while any(w.is_alive() for w in workers) and not stop.wait(1.0):
            pass
    except KeyboardInterrupt:
        pass
    print("\n종료 중...")
    for w in workers:
        w.stop()
    for w in workers:
        w.join(timeout=3)
    print("모든 카메라 종료")


def run_multi(sources: list[tuple], model_path: str, conf: float, frame_skip: int,
              device: str = "0", headless: bool = False):
    # 감지 기록 전에 테이블 보장 (웹 대시보드보다 먼저 떠도 동작하도록, 멱등)
    from src.database import sqlite_db
    sqlite_db.init_db()

    print(f"\n모델 로드 중: {model_path} (device={device})")
    detector = HighwayDetector(model_path, conf=conf, device=device)
    mode = "헤드리스" if headless else "종료: 'q'"
    print(f"카메라 {len(sources)}대 시작 ({mode})\n")

    workers = []
    for source, name in sources:
        w = CameraWorker(source, detector, name, frame_skip, headless=headless)
        workers.append(w)
        w.start()

    if headless:
        _run_headless(workers)
        return

    # 카메라별 최신 프레임 저장
    latest_frames = [None] * len(workers)
    names = [w.camera_name for w in workers]

    try:
        while any(w.is_alive() for w in workers):
            # 각 카메라에서 최신 프레임 수집
            for i, w in enumerate(workers):
                try:
                    latest_frames[i] = w.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            # 모든 카메라 프레임이 준비됐으면 그리드로 합쳐 표시
            ready = [f for f in latest_frames if f is not None]
            if ready:
                # 아직 안 온 카메라는 검은 화면으로 대체
                frames = [
                    f if f is not None else np.zeros((360, 640, 3), dtype=np.uint8)
                    for f in latest_frames
                ]
                grid = make_grid(frames, names)
                cv2.imshow("CCTV 모니터", grid)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n종료 중...")
                for w in workers:
                    w.stop()
                break

    except KeyboardInterrupt:
        print("\n종료 중...")
        for w in workers:
            w.stop()

    for w in workers:
        w.join(timeout=3)
    cv2.destroyAllWindows()
    print("모든 카메라 종료")


def _env_list(name: str) -> list[str]:
    """쉼표로 구분된 환경변수 → 리스트 (빈 항목 제거)"""
    return [v.strip() for v in os.getenv(name, "").split(",") if v.strip()]


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _default_device() -> str:
    env = os.getenv("DETECT_DEVICE", "").strip()
    if env:
        return env
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main():
    parser = argparse.ArgumentParser(description="다중 CCTV 실시간 모니터링")
    parser.add_argument("--sources", nargs="+",
                        help="영상 소스 목록 (예: video1.mp4 video2.mp4 rtsp://...). "
                             "생략 시 CAMERA_SOURCES 환경변수(쉼표 구분) 사용 — "
                             "RTSP 비밀번호가 프로세스 목록에 남지 않도록 환경변수 권장")
    parser.add_argument("--names", nargs="+",
                        help="카메라 이름 목록 (소스와 순서 동일). 생략 시 CAMERA_NAMES")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_PATH") or str(DEFAULT_MODEL))
    parser.add_argument("--conf", type=float, default=float(os.getenv("DETECT_CONF") or 0.25))
    parser.add_argument("--frame-skip", type=int, default=int(os.getenv("FRAME_SKIP") or 2),
                        help="N프레임마다 1번 추론 (기본 2, 카메라 많을수록 높게)")
    parser.add_argument("--device", type=str, default=None,
                        help="'0'=GPU, 'cpu'=CPU (기본: DETECT_DEVICE, 없으면 GPU 유무로 자동)")
    parser.add_argument("--headless", action="store_true", default=_env_flag("HEADLESS"),
                        help="화면/팝업 없이 실행 (서버, Docker). HEADLESS=1 과 같음")
    args = parser.parse_args()

    src_list = args.sources or _env_list("CAMERA_SOURCES")
    if not src_list:
        parser.error("--sources 또는 CAMERA_SOURCES 환경변수가 필요합니다")

    names = (args.names or _env_list("CAMERA_NAMES"))[:len(src_list)]
    if len(names) < len(src_list):
        names += [f"카메라 {i+1}" for i in range(len(names), len(src_list))]

    sources = []
    for src, name in zip(src_list, names):
        source = int(src) if src.isdigit() else src
        sources.append((source, name))

    run_multi(sources, args.model, args.conf, args.frame_skip,
              device=args.device or _default_device(), headless=args.headless)


if __name__ == "__main__":
    main()
