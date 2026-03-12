"""
prepare_data.py
===============
Step 1. ROI batch processing  →  ROI_output/ (128×128 grayscale PNG)
Step 2. Generate CCNet-format txt files  →  CCNet/data/

CCNet txt format (per line):
  <relative_path_from_CCNet_dir> <0-indexed_label>

Train/test split strategy (same as CCNet Tongji protocol):
  train.txt       : session1 only, persons 1-540  (label 0-539)
  test_gallery.txt: session1 only, persons 1-540  (registered gallery)
  test_probe.txt  : session2 only, persons 1-540  (closed-set probe)
  test_openset.txt: session2 only, persons 541-600 (unseen IDs for open-set eval)

Usage:
  python prepare_data.py            # run both steps
  python prepare_data.py --skip-roi # skip ROI step (if already done)
"""

import json
import argparse
from pathlib import Path
import sys

# ── Step 1: ROI processing ───────────────────────────────────────────────────
def run_roi(base: Path):
    print("=" * 60)
    print("STEP 1 — ROI Batch Processing (128×128 grayscale)")
    print("=" * 60)
    sys.path.insert(0, str(base))
    from ROI import process_dataset
    process_dataset(
        dataset_json = base / "dataset_split.json",
        output_dir   = base / "ROI_output",
        output_size  = 128,
        save_vis     = False,
    )

# ── Step 2: Generate CCNet txt files ────────────────────────────────────────
def generate_txt_files(base: Path):
    print("\n" + "=" * 60)
    print("STEP 2 — Generating CCNet data list files")
    print("=" * 60)

    with open(base / "dataset_split.json") as f:
        data = json.load(f)

    roi_root = base / "ROI_output"
    out_dir  = base / "CCNet" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Paths in txt are relative to CCNet/ directory
    # e.g.  ../ROI_output/train/0001/session1/00001.png  0
    def rel(path: Path) -> str:
        return "../" + str(path.relative_to(base)).replace("\\", "/")

    train_lines        = []
    test_gallery_lines = []
    test_probe_lines   = []
    test_openset_lines = []

    for person in data["persons"]:
        pid   = person["person_id"]
        sp    = person["split"]
        label = pid - 1  # 0-indexed

        s1_dir = roi_root / sp / f"{pid:04d}" / "session1"
        s2_dir = roi_root / sp / f"{pid:04d}" / "session2"

        if sp == "train":
            # train.txt: session1 only
            for fname in person["session1"]:
                p = s1_dir / (Path(fname).stem + ".png")
                train_lines.append(f"{rel(p)} {label}")

            # test_gallery.txt: session1, known IDs
            for fname in person["session1"]:
                p = s1_dir / (Path(fname).stem + ".png")
                test_gallery_lines.append(f"{rel(p)} {label}")

            # test_probe.txt: session2, known IDs (closed-set evaluation)
            for fname in person["session2"]:
                p = s2_dir / (Path(fname).stem + ".png")
                test_probe_lines.append(f"{rel(p)} {label}")

        else:  # test split (unseen persons 541-600)
            # test_openset.txt: session2, unknown IDs
            for fname in person["session2"]:
                p = s2_dir / (Path(fname).stem + ".png")
                test_openset_lines.append(f"{rel(p)} {label}")

    files = {
        "train_ours.txt":        train_lines,
        "test_gallery.txt":      test_gallery_lines,
        "test_probe.txt":        test_probe_lines,
        "test_openset.txt":      test_openset_lines,
    }

    for fname, lines in files.items():
        out_path = out_dir / fname
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  {fname:30s}  {len(lines):>6} lines  →  {out_path}")

    print("\nSummary:")
    print(f"  Train persons   : 540  (persons 1-540, session1 only)")
    print(f"  Gallery         : 540 persons × 10 images = {len(test_gallery_lines)}")
    print(f"  Probe (closed)  : 540 persons × 10 images = {len(test_probe_lines)}")
    print(f"  Probe (open set):  60 persons × 10 images = {len(test_openset_lines)}")
    print(f"  num_classes for training: 540")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-roi", action="store_true",
                        help="Skip ROI processing (use existing ROI_output/)")
    args = parser.parse_args()

    base = Path(__file__).parent

    if not args.skip_roi:
        run_roi(base)
    else:
        print("Skipping ROI processing.")

    generate_txt_files(base)

    print("\n" + "=" * 60)
    print("All done! Ready for cloud training.")
    print("=" * 60)
    print("\nCloud training command:")
    print("  cd CCNet")
    print("  python train.py \\")
    print("    --id_num 540 \\")
    print("    --train_set_file ./data/train_ours.txt \\")
    print("    --test_set_file  ./data/test_probe.txt \\")
    print("    --batch_size 512 \\")
    print("    --epoch_num 3000 \\")
    print("    --lr 0.001")
