"""
YOLOv8 학습 스크립트
"""

import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"


def check_environment():
    """GPU/CUDA 환경 확인"""
    print("[환경 확인]")
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        print(f"  CUDA: {torch.version.cuda}")
    else:
        print("  GPU: 사용 불가 (CPU 모드)")
    print()


def train(config_path: str = None):
    """학습 실행"""
    check_environment()

    # 설정 로드
    config_path = Path(config_path or CONFIG_PATH)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"[학습 설정]")
    print(f"  모델: {cfg['model']}")
    print(f"  에폭: {cfg['epochs']}")
    print(f"  배치: {cfg['batch']}")
    print(f"  이미지 크기: {cfg['imgsz']}")
    print(f"  디바이스: GPU {cfg['device']}")
    print()

    # 데이터셋 경로를 절대 경로로 변환
    data_yaml = PROJECT_ROOT / cfg["data"]
    if not data_yaml.exists():
        print(f"[오류] dataset.yaml을 찾을 수 없습니다: {data_yaml}")
        print("  prepare_dataset.py 를 먼저 실행하세요.")
        return

    # 모델 로드 (사전학습 가중치 사용)
    model = YOLO(cfg["model"])
    print(f"모델 로드 완료: {cfg['model']}")

    # 학습 시작
    print("\n학습 시작...")
    results = model.train(
        data=str(data_yaml),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg["device"],
        workers=cfg.get("workers", 4),
        project=str(PROJECT_ROOT / cfg.get("project", "runs/train")),
        name=cfg.get("name", "highway_detector"),
        exist_ok=cfg.get("exist_ok", True),
        pretrained=cfg.get("pretrained", True),
        optimizer=cfg.get("optimizer", "AdamW"),
        lr0=cfg.get("lr0", 0.001),
        weight_decay=cfg.get("weight_decay", 0.0005),
        patience=cfg.get("patience", 20),
        save_period=cfg.get("save_period", 10),
        val=cfg.get("val", True),
        plots=cfg.get("plots", True),
        verbose=True,
    )

    # 결과 저장 경로
    save_dir = Path(results.save_dir)
    best_model = save_dir / "weights" / "best.pt"

    print(f"\n학습 완료!")
    print(f"  결과 폴더: {save_dir}")
    print(f"  최적 모델: {best_model}")

    # models/ 폴더에 복사
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    if best_model.exists():
        import shutil
        dest = models_dir / "highway_detector_best.pt"
        shutil.copy2(best_model, dest)
        print(f"  모델 복사: {dest}")

    return results


def resume_training(checkpoint: str):
    """중단된 학습 재개"""
    model = YOLO(checkpoint)
    print(f"체크포인트에서 재개: {checkpoint}")
    results = model.train(resume=True)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 학습")
    parser.add_argument("--config", type=str, default=None,
                        help="학습 설정 파일 경로")
    parser.add_argument("--resume", type=str, default=None,
                        help="체크포인트 경로 (학습 재개)")
    args = parser.parse_args()

    if args.resume:
        resume_training(args.resume)
    else:
        train(args.config)
