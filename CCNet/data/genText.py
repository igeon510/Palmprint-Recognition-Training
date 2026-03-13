"""
genText.py  —  우리 데이터셋용 txt 파일 생성기
클라우드 환경에서 ROI_output 경로가 달라질 때 여기서 재생성.

Usage (CCNet/ 디렉토리에서):
  python data/genText.py --roi_root /absolute/path/to/ROI_output

생성 파일:
  train_ours.txt    : session1, persons 1-540  (train)
  test_gallery.txt  : session1, persons 1-540  (gallery for matching)
  test_probe.txt    : session2, persons 1-540  (closed-set probe)
  test_openset.txt  : session2, persons 541-600 (open-set, unseen IDs)
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roi_root", type=str, default="../ROI_output",
        help="Path to ROI_output directory (default: ../ROI_output)"
    )
    parser.add_argument(
        "--split_json", type=str, default="../dataset_split.json",
        help="Path to dataset_split.json"
    )
    args = parser.parse_args()

    roi_root   = Path(args.roi_root)
    split_json = Path(args.split_json)
    out_dir    = Path(__file__).parent

    with open(split_json) as f:
        data = json.load(f)

    train_lines        = []
    test_gallery_lines = []
    test_probe_lines   = []
    test_openset_lines = []

    for person in data["persons"]:
        pid   = person["person_id"]
        sp    = person["split"]
        label = pid - 1   # 0-indexed

        s1_dir = roi_root / sp / f"{pid:04d}" / "session1"
        s2_dir = roi_root / sp / f"{pid:04d}" / "session2"

        if sp == "train":
            for fname in person["session1"]:
                p = s1_dir / (Path(fname).stem + ".png")
                train_lines.append(f"{p} {label}")
                test_gallery_lines.append(f"{p} {label}")
            for fname in person["session2"]:
                p = s2_dir / (Path(fname).stem + ".png")
                train_lines.append(f"{p} {label}")
                test_probe_lines.append(f"{p} {label}")
        else:
            for fname in person["session2"]:
                p = s2_dir / (Path(fname).stem + ".png")
                test_openset_lines.append(f"{p} {label}")

    files = {
        "train_ours.txt":   train_lines,
        "test_gallery.txt": test_gallery_lines,
        "test_probe.txt":   test_probe_lines,
        "test_openset.txt": test_openset_lines,
    }
    for fname, lines in files.items():
        out_path = out_dir / fname
        out_path.write_text("\n".join(lines) + "\n")
        print(f"  {fname:30s}  {len(lines):>6} lines  →  {out_path}")


if __name__ == "__main__":
    main()
