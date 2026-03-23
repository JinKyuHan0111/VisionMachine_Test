"""
공개 데이터셋 다운로드 스크립트
- D-Fire: 화재/연기 감지 데이터셋
- Roboflow Universe: 사고/교통 데이터셋
"""

import os
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """파일 다운로드 (진행률 표시)"""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f, tqdm(
        desc=desc, total=total, unit="B", unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def download_dfire():
    """
    D-Fire Dataset (화재/연기)
    출처: https://github.com/gaiasd/DFireDataset
    약 21,000장 이미지 포함
    """
    print("\n[1/2] D-Fire Dataset 다운로드 중...")
    dest_dir = DATA_DIR / "raw" / "dfire"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # D-Fire는 GitHub에서 직접 clone 권장
    print("D-Fire 데이터셋은 Git LFS가 필요합니다.")
    print("아래 명령어를 터미널에서 실행하세요:\n")
    print("  git clone https://github.com/gaiasd/DFireDataset.git data/raw/dfire")
    print("\n또는 Roboflow에서 다운로드:")
    print("  https://universe.roboflow.com/school-tvtyq/d-fire\n")


def download_from_roboflow(api_key: str, workspace: str, project: str,
                            version: int, dest_dir: Path):
    """Roboflow에서 데이터셋 다운로드"""
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8", location=str(dest_dir))
        print(f"다운로드 완료: {dest_dir}")
        return dataset
    except ImportError:
        print("roboflow 패키지가 없습니다. pip install roboflow")
        return None


def setup_roboflow_datasets(api_key: str = None):
    """
    Roboflow Universe 추천 데이터셋
    API 키는 https://roboflow.com 에서 무료 발급
    """
    if not api_key:
        print("\n[Roboflow 데이터셋 안내]")
        print("아래 데이터셋들을 Roboflow Universe에서 YOLOv8 형식으로 다운로드하세요:\n")

        datasets = [
            {
                "name": "화재/연기 감지",
                "url": "https://universe.roboflow.com/nuri-gjsug/fire-and-smoke-detection-bkxl5",
                "desc": "고속도로 화재 및 연기 감지"
            },
            {
                "name": "차량 사고 감지",
                "url": "https://universe.roboflow.com/accident-detection-ffdkg/accident-detection-8dvh5",
                "desc": "교통사고 감지"
            },
            {
                "name": "고속도로 차량",
                "url": "https://universe.roboflow.com/prjct-3gqly/highway-vehicle-detection",
                "desc": "고속도로 차량 종류 분류"
            },
        ]

        for ds in datasets:
            print(f"  [{ds['name']}]")
            print(f"   설명: {ds['desc']}")
            print(f"   URL:  {ds['url']}\n")

        print("다운로드 후 data/raw/ 폴더에 넣고 prepare_dataset.py 를 실행하세요.")
        return

    # API 키가 있으면 자동 다운로드
    download_from_roboflow(
        api_key=api_key,
        workspace="nuri-gjsug",
        project="fire-and-smoke-detection-bkxl5",
        version=1,
        dest_dir=DATA_DIR / "raw" / "fire_smoke"
    )


def main():
    print("=" * 50)
    print("  Highway CCTV 데이터셋 다운로드")
    print("=" * 50)

    download_dfire()

    # Roboflow API 키가 있으면 아래 주석 해제
    # api_key = os.getenv("ROBOFLOW_API_KEY", "")
    # setup_roboflow_datasets(api_key)

    setup_roboflow_datasets()

    print("\n완료! 데이터 준비 후 prepare_dataset.py 를 실행하세요.")


if __name__ == "__main__":
    main()
