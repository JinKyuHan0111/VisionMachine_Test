"""
D-Fire 데이터셋 다운로드 및 기존 Roboflow 데이터와 병합
D-Fire: https://github.com/gaiasd/DFireDataset
클래스: 0=fire, 1=smoke (현재 구조와 동일 → 변환 불필요)
"""

import zipfile
import shutil
import random
from pathlib import Path

# ─── 경로 설정 ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DFIRE_DIR    = PROJECT_ROOT / "data" / "raw" / "dfire"
MERGED_DIR   = PROJECT_ROOT / "data" / "raw" / "merged"
ROBOFLOW_DIRS = [
    PROJECT_ROOT / "data" / "raw" / "fire_smoke",
    PROJECT_ROOT / "data" / "raw" / "fire_smoke_mx4z8",
]

VAL_RATIO  = 0.15
TEST_RATIO = 0.10


# ─── 1. D-Fire ZIP 압축 해제 ─────────────────────────────────
def extract_dfire():
    """data/raw/dfire/D-Fire.zip 압축 해제"""
    DFIRE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DFIRE_DIR / "D-Fire.zip"

    if not zip_path.exists():
        print("[오류] D-Fire.zip 파일이 없습니다.")
        print()
        print("  다운로드 방법:")
        print("  1. https://github.com/gaiasd/DFireDataset 접속")
        print("  2. ZIP 파일 다운로드")
        print(f"  3. 아래 경로에 저장:")
        print(f"     {zip_path}")
        print()
        print("  저장 후 스크립트를 다시 실행하세요.")
        return None

    extract_dir = DFIRE_DIR / "extracted"
    if extract_dir.exists():
        print("[건너뜀] 이미 압축 해제됨")
        return extract_dir

    print(f"[압축 해제] {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for i, member in enumerate(members):
            # 경로 조작 방지: ../ 포함된 항목 건너뜀
            member_path = Path(member.filename)
            if ".." in member_path.parts:
                print(f"  [보안] 건너뜀: {member.filename}")
                continue
            target = extract_dir / member_path
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            if i % 1000 == 0:
                print(f"\r  {i}/{len(members)}...", end="", flush=True)
    print()
    print(f"  → {extract_dir}")
    return extract_dir


# ─── 2. D-Fire 구조 파악 ──────────────────────────────────────
def find_dfire_pairs(extract_dir: Path) -> list[tuple[Path, Path]]:
    """(image_path, label_path) 쌍 목록 반환"""
    pairs = []
    for img in extract_dir.rglob("*.jpg"):
        label = img.parent.parent / "labels" / img.with_suffix(".txt").name
        if not label.exists():
            # images/ → labels/ 구조가 아닌 경우 같은 폴더 시도
            label = img.with_suffix(".txt")
        if label.exists():
            pairs.append((img, label))
    print(f"  D-Fire 유효 샘플: {len(pairs)}장")
    return pairs


# ─── 3. Roboflow 데이터 수집 ──────────────────────────────────
def collect_roboflow_pairs() -> list[tuple[Path, Path]]:
    pairs = []
    for dataset_dir in ROBOFLOW_DIRS:
        count = 0
        for split in ("train", "valid", "test"):
            img_dir   = dataset_dir / split / "images"
            label_dir = dataset_dir / split / "labels"
            if not img_dir.exists():
                continue
            for img in img_dir.glob("*.jpg"):
                label = label_dir / img.with_suffix(".txt").name
                if label.exists():
                    pairs.append((img, label))
                    count += 1
        print(f"  {dataset_dir.name}: {count}장")
    print(f"  Roboflow 합계: {len(pairs)}장")
    return pairs


# ─── 4. 병합 및 분할 ──────────────────────────────────────────
def merge_and_split(rf_pairs, df_pairs):
    all_pairs = rf_pairs + df_pairs
    random.seed(42)
    random.shuffle(all_pairs)

    n       = len(all_pairs)
    n_val   = int(n * VAL_RATIO)
    n_test  = int(n * TEST_RATIO)
    n_train = n - n_val - n_test

    splits = {
        "train": all_pairs[:n_train],
        "valid": all_pairs[n_train:n_train + n_val],
        "test":  all_pairs[n_train + n_val:],
    }

    print(f"\n[분할 결과]")
    print(f"  전체: {n}장")
    print(f"  train: {n_train} / val: {n_val} / test: {n_test}")

    for split, pairs in splits.items():
        img_out   = MERGED_DIR / split / "images"
        label_out = MERGED_DIR / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for i, (img, label) in enumerate(pairs):
            # 파일명 중복 방지: 인덱스 prefix 추가
            stem = f"{split}_{i:06d}"
            shutil.copy2(img,   img_out   / (stem + ".jpg"))
            shutil.copy2(label, label_out / (stem + ".txt"))

        print(f"  {split}: {len(pairs)}장 복사 완료")


# ─── 5. dataset.yaml 업데이트 ────────────────────────────────
def update_dataset_yaml():
    yaml_path = PROJECT_ROOT / "configs" / "dataset.yaml"
    content = f"""# Highway CCTV - Fire Detection Dataset Config (merged: Roboflow x2 + D-Fire)
# YOLOv8 format

path: {MERGED_DIR.as_posix()}
train: train/images
val: valid/images
test: test/images

# Classes (flame을 fire로 병합)
nc: 2
names:
  0: fire
  1: smoke
"""
    yaml_path.write_text(content, encoding="utf-8")
    print(f"\n[완료] configs/dataset.yaml → merged 데이터셋 경로로 업데이트")


# ─── 메인 ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  D-Fire 다운로드 + 데이터셋 병합")
    print("=" * 50)

    print("\n[1/4] D-Fire ZIP 압축 해제...")
    extract_dir = extract_dfire()

    print("\n[2/4] D-Fire 샘플 수집...")
    df_pairs = find_dfire_pairs(extract_dir)

    print("\n[3/4] Roboflow 샘플 수집...")
    rf_pairs = collect_roboflow_pairs()

    print("\n[4/4] 병합 및 분할...")
    merge_and_split(rf_pairs, df_pairs)

    update_dataset_yaml()

    print("\n완료! 이제 학습을 시작하세요:")
    print("  python src/training/train.py")
