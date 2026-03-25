"""
데스크톱 팝업 경고 시스템
위험 상황 감지 시 화면에 팝업창 표시
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
from datetime import datetime
from pathlib import Path


# 클래스별 위험도 설정
ALERT_CONFIG = {
    "accident": {
        "label": "차량 사고",
        "level": "CRITICAL",
        "color": "#FF0000",
        "bg": "#FFE0E0",
        "sound": True,
    },
    "fire": {
        "label": "화재/산불",
        "level": "CRITICAL",
        "color": "#FF4500",
        "bg": "#FFF0E0",
        "sound": True,
    },
    "flame": {
        "label": "화염 감지",
        "level": "CRITICAL",
        "color": "#FF4500",
        "bg": "#FFF0E0",
        "sound": True,
    },
    "smoke": {
        "label": "연기 감지",
        "level": "WARNING",
        "color": "#FF8C00",
        "bg": "#FFFAE0",
        "sound": False,
    },
    "stopped_vehicle": {
        "label": "갓길 정차",
        "level": "INFO",
        "color": "#1E90FF",
        "bg": "#E0F0FF",
        "sound": False,
    },
    "debris": {
        "label": "낙하물 감지",
        "level": "WARNING",
        "color": "#FF8C00",
        "bg": "#FFFAE0",
        "sound": False,
    },
}

# 중복 알림 방지용 쿨다운 (초)
COOLDOWN = {
    "CRITICAL": 10,
    "WARNING": 30,
    "INFO": 60,
}

_last_alert_time: dict = {}
_alert_lock = threading.Lock()


class AlertWindow:
    """팝업 경고창"""

    def __init__(self, class_name: str, confidence: float,
                 location: str = "카메라 1", frame_path: str = None):
        self.class_name = class_name
        self.confidence = confidence
        self.location = location
        self.frame_path = frame_path
        self.config = ALERT_CONFIG.get(class_name, {
            "label": class_name,
            "level": "WARNING",
            "color": "#FF8C00",
            "bg": "#FFFAE0",
            "sound": False,
        })

    def show(self):
        """팝업창 표시 (별도 스레드에서 실행)"""
        root = tk.Tk()
        root.withdraw()  # 메인 창 숨김

        popup = tk.Toplevel(root)
        self._build_popup(popup)
        popup.mainloop()

    def _build_popup(self, popup: tk.Toplevel):
        """팝업 UI 구성"""
        cfg = self.config
        level = cfg["level"]
        color = cfg["color"]
        bg = cfg["bg"]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 창 설정
        popup.title(f"[{level}] 위험 상황 감지")
        popup.geometry("450x280")
        popup.resizable(False, False)
        popup.configure(bg=bg)
        popup.attributes("-topmost", True)  # 항상 최상단
        popup.attributes("-alpha", 0.95)

        # 화면 오른쪽 하단에 배치
        screen_w = popup.winfo_screenwidth()
        screen_h = popup.winfo_screenheight()
        popup.geometry(f"+{screen_w - 470}+{screen_h - 320}")

        # 헤더 (색상 바)
        header = tk.Frame(popup, bg=color, height=8)
        header.pack(fill="x")

        # 메인 컨텐츠
        content = tk.Frame(popup, bg=bg, padx=20, pady=15)
        content.pack(fill="both", expand=True)

        # 레벨 뱃지
        level_colors = {"CRITICAL": "#FF0000", "WARNING": "#FF8C00", "INFO": "#1E90FF"}
        badge_bg = level_colors.get(level, "#888")
        badge = tk.Label(content, text=f" {level} ",
                         font=("Arial", 10, "bold"),
                         fg="white", bg=badge_bg,
                         padx=6, pady=2)
        badge.pack(anchor="w")

        # 제목
        title_label = tk.Label(
            content,
            text=f"⚠  {cfg['label']} 감지",
            font=("Arial", 18, "bold"),
            fg=color, bg=bg
        )
        title_label.pack(pady=(8, 4))

        # 구분선
        ttk.Separator(content, orient="horizontal").pack(fill="x", pady=6)

        # 상세 정보
        info_frame = tk.Frame(content, bg=bg)
        info_frame.pack(fill="x")

        def info_row(label, value):
            row = tk.Frame(info_frame, bg=bg)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"{label}:", width=8, anchor="w",
                     font=("Arial", 10), fg="#555", bg=bg).pack(side="left")
            tk.Label(row, text=value, anchor="w",
                     font=("Arial", 10, "bold"), fg="#222", bg=bg).pack(side="left")

        info_row("위치", self.location)
        info_row("신뢰도", f"{self.confidence:.1%}")
        info_row("시각", timestamp)

        # 버튼 영역
        btn_frame = tk.Frame(popup, bg=bg, padx=20, pady=10)
        btn_frame.pack(fill="x")

        def on_close():
            popup.destroy()

        close_btn = tk.Button(
            btn_frame,
            text="확인",
            font=("Arial", 11, "bold"),
            bg=color, fg="white",
            activebackground="#CC0000",
            relief="flat",
            padx=20, pady=6,
            cursor="hand2",
            command=on_close
        )
        close_btn.pack(side="right")

        # 5초 후 자동 닫힘 카운트다운
        countdown_label = tk.Label(
            btn_frame,
            text="5초 후 자동 닫힘",
            font=("Arial", 9),
            fg="#888", bg=bg
        )
        countdown_label.pack(side="right", padx=(0, 10))

        def countdown(n):
            if n > 0:
                countdown_label.config(text=f"{n}초 후 자동 닫힘")
                popup.after(1000, countdown, n - 1)
            else:
                popup.destroy()

        popup.after(1000, countdown, 4)

        # 깜빡임 효과 (CRITICAL)
        if level == "CRITICAL":
            self._blink_title(popup, title_label, color, bg)

    def _blink_title(self, popup, label, color1, color2, count=6):
        """타이틀 깜빡임 효과"""
        if count <= 0:
            label.config(fg=color1)
            return
        current = label.cget("fg")
        next_color = color2 if current == color1 else color1
        label.config(fg=next_color)
        popup.after(400, self._blink_title, popup, label, color1, color2, count - 1)


def trigger_alert(class_name: str, confidence: float,
                  location: str = "카메라 1", frame_path: str = None,
                  use_db: bool = True):
    """
    경고 팝업 트리거 (스레드 안전, 쿨다운 적용, DB 기록)

    Args:
        class_name: 감지된 클래스명 (fire, flame, smoke 등)
        confidence: 신뢰도 (0.0 ~ 1.0)
        location: 감지 위치 (카메라 이름)
        frame_path: 캡처 프레임 경로 (옵션)
        use_db: DB 기록 여부 (기본 True)
    """
    cfg = ALERT_CONFIG.get(class_name, {"level": "WARNING"})
    level = cfg["level"]
    cooldown = COOLDOWN.get(level, 30)

    with _alert_lock:
        key = f"{class_name}_{location}"
        now = time.time()
        if key in _last_alert_time:
            if now - _last_alert_time[key] < cooldown:
                return  # 쿨다운 중
        _last_alert_time[key] = now

    # DB 기록 (db_manager 통해 SQLite + ChromaDB 동시 저장)
    if use_db:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from src.database import db_manager
            db_manager.record_detection(
                class_name=class_name,
                confidence=confidence,
                source=location,
                frame_path=frame_path,
            )
        except Exception as e:
            print(f"[DB 기록 실패] {e}")

    # 별도 스레드에서 팝업 표시
    alert = AlertWindow(class_name, confidence, location, frame_path)
    t = threading.Thread(target=alert.show, daemon=True)
    t.start()


# ── 테스트 ────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    print("경고 팝업 테스트...")
    trigger_alert("fire", 0.92, "고속도로 1구간 카메라 3")
    time.sleep(0.5)
    trigger_alert("accident", 0.87, "고속도로 2구간 카메라 1")
    time.sleep(0.5)
    trigger_alert("smoke", 0.75, "고속도로 1구간 카메라 5")

    input("팝업 확인 후 Enter 키를 누르세요...")
