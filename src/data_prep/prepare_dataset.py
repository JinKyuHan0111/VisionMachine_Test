"""
데이터셋 전처리 및 train/val/test 분리 스크립트
"""

import shutil
import random
import yaml
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "datasets"


def split_dataset(images_dir: Path, labels_dir: Path,
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.2,
                  test_ratio: float = 0.1,
                  seed: int = 42):
    """
    이미지/라벨 쌍을 train/val/test로 분리

    Args:
        images_dir: 이미지 폴더 (jpg/png)
        labels_dir: 라벨 폴더 (YOLO .txt 파일)
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        seed: 랜덤 시드
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "비율의 합이 1이어야 합니다"

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in images_dir.iterdir()
              if f.suffix.lower() in image_exts]

    # 라벨 파일이 있는 이미지만 필터링
    valid_pairs = []
    for img in images:
        label = labels_dir / (img.stem + ".txt")
        if label.exists():
            valid_pairs.append((img, label))

    print(f"유효한 이미지/라벨 쌍: {len(valid_pairs)}개")

    random.seed(seed)
    random.shuffle(valid_pairs)

    n = len(valid_pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": valid_pairs[:n_train],
        "val": valid_pairs[n_train:n_train + n_val],
        "test": valid_pairs[n_train + n_val:],
    }

    for split_name, pairs in splits.items():
        img_out = DATASET_DIR / "images" / split_name
        lbl_out = DATASET_DIR / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in tqdm(pairs, desc=f"{split_name} 복사 중"):
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

        print(f"  {split_name}: {len(pairs)}개")

    print(f"\n데이터셋 준비 완료: {DATASET_DIR}")


def merge_roboflow_datasets(source_dirs: list, dest_dir: Path = None):
    """
    여러 Roboflow 다운로드 폴더를 하나로 합치기
    Roboflow YOLOv8 형식: {root}/train/images, {root}/train/labels 구조
    """
    if dest_dir is None:
        dest_dir = DATA_DIR / "merged"

    for split in ["train", "val", "test"]:
        (dest_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dest_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    total = 0
    for src in source_dirs:
        src = Path(src)
        for split in ["train", "val", "test"]:
            src_img = src / split / "images"
            src_lbl = src / split / "labels"
            if not src_img.exists():
                continue

            for img in src_img.glob("*"):
                dest = dest_dir / split / "images" / img.name
                shutil.copy2(img, dest)
                total += 1

            if src_lbl.exists():
                for lbl in src_lbl.glob("*.txt"):
                    dest = dest_dir / split / "labels" / lbl.name
                    shutil.copy2(lbl, dest)

    print(f"병합 완료: 총 {total}개 이미지 -> {dest_dir}")
    return dest_dir


def update_dataset_yaml(classes: dict, dataset_dir: Path = None):
    """dataset.yaml 파일 업데이트"""
    if dataset_dir is None:
        dataset_dir = DATASET_DIR

    config_path = PROJECT_ROOT / "configs" / "dataset.yaml"
    config = {
        "path": str(dataset_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": classes,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"dataset.yaml 업데이트 완료: {len(classes)}개 클래스")


def check_dataset():
    """데이터셋 상태 확인"""
    print("\n[데이터셋 현황]")
    for split in ["train", "val", "test"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        n_img = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"  {split:6s}: 이미지 {n_img:5d}개, 라벨 {n_lbl:5d}개")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="데이터셋 준비")
    parser.add_argument("--images", type=str,
                        default=str(DATA_DIR / "images"),
                        help="이미지 폴더 경로")
    parser.add_argument("--labels", type=str,
                        default=str(DATA_DIR / "labels"),
                        help="라벨 폴더 경로")
    parser.add_argument("--check", action="store_true",
                        help="현재 데이터셋 현황만 확인")
    args = parser.parse_args()

    if args.check:
        check_dataset()
    else:
        split_dataset(Path(args.images), Path(args.labels))
        check_dataset()
