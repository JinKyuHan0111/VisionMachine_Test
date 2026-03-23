"""
동영상에서 프레임 추출 스크립트
CCTV 영상 파일 -> 이미지 프레임 추출
"""

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def extract_frames(video_path: str, output_dir: str,
                   fps: int = 2, max_frames: int = None):
    """
    동영상에서 프레임 추출

    Args:
        video_path: 입력 동영상 경로
        output_dir: 출력 이미지 폴더
        fps: 초당 추출할 프레임 수 (기본 2fps)
        max_frames: 최대 추출 프레임 수 (None=무제한)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[오류] 동영상을 열 수 없습니다: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))

    print(f"동영상: {video_path.name}")
    print(f"  원본 FPS: {video_fps:.1f}, 추출 간격: 매 {frame_interval}프레임")
    print(f"  총 프레임: {total_frames}")

    count = 0
    saved = 0
    stem = video_path.stem

    with tqdm(total=total_frames, desc="추출 중") as bar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_interval == 0:
                filename = output_dir / f"{stem}_frame_{count:06d}.jpg"
                cv2.imwrite(str(filename), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
                if max_frames and saved >= max_frames:
                    break

            count += 1
            bar.update(1)

    cap.release()
    print(f"추출 완료: {saved}장 -> {output_dir}")
    return saved


def batch_extract(video_dir: str, output_dir: str, fps: int = 2):
    """폴더 내 모든 동영상에서 프레임 추출"""
    video_dir = Path(video_dir)
    extensions = [".mp4", ".avi", ".mov", ".mkv", ".ts"]
    videos = [f for f in video_dir.iterdir()
              if f.suffix.lower() in extensions]

    if not videos:
        print(f"[오류] {video_dir}에 동영상 파일이 없습니다.")
        return

    print(f"총 {len(videos)}개 동영상 발견")
    total = 0
    for video in videos:
        saved = extract_frames(str(video), output_dir, fps=fps)
        total += saved

    print(f"\n전체 추출 완료: {total}장")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="동영상 프레임 추출")
    parser.add_argument("--video", type=str, help="단일 동영상 경로")
    parser.add_argument("--video_dir", type=str, help="동영상 폴더 경로")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "data" / "images" / "raw_frames"),
                        help="출력 폴더")
    parser.add_argument("--fps", type=int, default=2, help="초당 추출 프레임 수")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="최대 추출 프레임 수")
    args = parser.parse_args()

    if args.video:
        extract_frames(args.video, args.output, args.fps, args.max_frames)
    elif args.video_dir:
        batch_extract(args.video_dir, args.output, args.fps)
    else:
        print("사용법: python extract_frames.py --video <영상경로>")
        print("        python extract_frames.py --video_dir <폴더경로>")
