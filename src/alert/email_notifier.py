"""
Gmail SMTP 이메일 알림
화재/연기 감지 시 이메일 발송
"""

import smtplib
import threading
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
from datetime import datetime

# ─── 설정 (config.py 또는 환경변수로 관리) ───────────────────
EMAIL_CONFIG = {
    "sender":    "",          # 보내는 Gmail 주소
    "password":  "",          # Gmail 앱 비밀번호 (16자리)
    "recipient": "",          # 받는 이메일 주소
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "enabled":   False,       # setup() 호출 후 True로 변경
}

# 쿨다운: 같은 클래스는 N초에 1번만 발송
EMAIL_COOLDOWN = 300  # 5분
_last_sent: dict = {}
_lock = threading.Lock()


def setup(sender: str, password: str, recipient: str):
    """이메일 설정 초기화"""
    EMAIL_CONFIG["sender"]    = sender
    EMAIL_CONFIG["password"]  = password
    EMAIL_CONFIG["recipient"] = recipient
    EMAIL_CONFIG["enabled"]   = True
    print(f"[이메일 알림] 설정 완료: {sender} → {recipient}")


def _build_html(class_name: str, confidence: float, location: str, timestamp: str) -> str:
    color = "#FF4500" if class_name == "fire" else "#888888"
    label = "화재 감지" if class_name == "fire" else "연기 감지"
    level = "CRITICAL" if class_name == "fire" else "WARNING"
    return f"""
    <html><body style="font-family:sans-serif; background:#f5f5f5; padding:20px;">
      <div style="max-width:480px; margin:0 auto; background:#fff;
                  border-radius:10px; overflow:hidden; box-shadow:0 2px 8px #0002;">
        <div style="background:{color}; padding:20px; color:#fff;">
          <h2 style="margin:0;">⚠ {label}</h2>
          <p style="margin:4px 0 0; opacity:0.85;">{level}</p>
        </div>
        <div style="padding:24px;">
          <table style="width:100%; border-collapse:collapse;">
            <tr><td style="color:#888; padding:6px 0;">감지 유형</td>
                <td style="font-weight:600;">{class_name.upper()}</td></tr>
            <tr><td style="color:#888; padding:6px 0;">신뢰도</td>
                <td style="font-weight:600;">{confidence*100:.1f}%</td></tr>
            <tr><td style="color:#888; padding:6px 0;">카메라</td>
                <td>{location}</td></tr>
            <tr><td style="color:#888; padding:6px 0;">감지 시각</td>
                <td>{timestamp}</td></tr>
          </table>
        </div>
        <div style="padding:12px 24px; background:#f9f9f9; color:#aaa; font-size:12px;">
          VisionMachine 화재 감지 시스템
        </div>
      </div>
    </body></html>
    """


def send_alert(class_name: str, confidence: float,
               location: str = "카메라 1", frame_path: str = None):
    """
    이메일 경고 발송 (별도 스레드, 쿨다운 적용)

    Args:
        class_name:  감지 클래스 (fire, smoke)
        confidence:  신뢰도
        location:    카메라 이름
        frame_path:  캡처 이미지 경로 (첨부, 선택)
    """
    if not EMAIL_CONFIG["enabled"]:
        return

    with _lock:
        now = time.time()
        if class_name in _last_sent and now - _last_sent[class_name] < EMAIL_COOLDOWN:
            return
        _last_sent[class_name] = now

    t = threading.Thread(
        target=_send, args=(class_name, confidence, location, frame_path), daemon=True
    )
    t.start()


def _send(class_name: str, confidence: float, location: str, frame_path: str | None):
    """실제 이메일 발송"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        label = "화재 감지" if class_name == "fire" else "연기 감지"

        msg = MIMEMultipart("related")
        msg["From"]    = EMAIL_CONFIG["sender"]
        msg["To"]      = EMAIL_CONFIG["recipient"]
        msg["Subject"] = f"[VisionMachine] {label} - {location} ({timestamp})"

        # HTML 본문
        html = _build_html(class_name, confidence, location, timestamp)
        msg.attach(MIMEText(html, "html", "utf-8"))

        # 캡처 이미지 첨부
        if frame_path:
            img_path = Path(frame_path)
            if img_path.exists():
                with open(img_path, "rb") as f:
                    img = MIMEImage(f.read(), name=img_path.name)
                    img.add_header("Content-Disposition", "attachment",
                                   filename=img_path.name)
                    msg.attach(img)

        with smtplib.SMTP(EMAIL_CONFIG["smtp_host"], EMAIL_CONFIG["smtp_port"]) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["sender"], EMAIL_CONFIG["password"])
            server.send_message(msg)

        print(f"[이메일] 발송 완료 → {EMAIL_CONFIG['recipient']} ({class_name})")

    except Exception as e:
        print(f"[이메일] 발송 실패: {e}")
