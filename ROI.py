"""
Palm Line ROI Extractor
=======================
Extracts the central palm region (containing all major palm lines) from
any input image — real phone photos or controlled dataset scans.

Pipeline
--------
1. MediaPipe Hands  →  21 landmarks  →  rotation-corrected square crop  (primary)
2. Contour-based fallback  →  for dark-background dataset images where MediaPipe fails

Landmarks used
--------------
  0  : wrist
  5  : index MCP  (base of index finger)
  9  : middle MCP
 13  : ring MCP
 17  : pinky MCP
The quadrilateral [5→17, 0] spans the entire palm-line region.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional
import json


# ────────────────────────────────────────────────────────────────────────────
class PalmROIExtractor:
    """
    Extract a normalised palm-line ROI from any hand image.

    Parameters
    ----------
    output_size : int
        Side length (pixels) of the square output crop.
    grayscale : bool
        If True, return a single-channel grayscale image.
    min_detection_confidence : float
        MediaPipe detection threshold (lower = more permissive, useful for
        dark/scanner-type dataset images).
    """

    def __init__(
        self,
        output_size: int = 128,
        grayscale: bool = True,
        min_detection_confidence: float = 0.3,
    ):
        self.output_size = output_size
        self.grayscale = grayscale
        self.min_detection_confidence = min_detection_confidence
        self._mp_hands = mp.solutions.hands

    # ------------------------------------------------------------------ API
    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract palm ROI from a BGR (OpenCV) image.

        Returns
        -------
        np.ndarray  shape (output_size, output_size) if grayscale else (H, W, 3)
        None        if extraction failed
        """
        roi = self._extract_mediapipe(image)
        if roi is None:
            roi = self._extract_contour(image)
        return roi

    def extract_from_path(self, path: str | Path) -> Optional[np.ndarray]:
        image = cv2.imread(str(path))
        if image is None:
            return None
        return self.extract(image)

    # ─────────────────────────────────────── primary: MediaPipe landmark crop
    def _extract_mediapipe(self, image: np.ndarray) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with self._mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
        ) as hands:
            results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        lm = results.multi_hand_landmarks[0].landmark

        def pt(idx) -> np.ndarray:
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

        wrist      = pt(0)
        idx_mcp    = pt(5)
        mid_mcp    = pt(9)
        pinky_mcp  = pt(17)

        # ── Palm center: 50% between wrist and finger-base midline
        #    Centering here captures all three main lines:
        #    heart line (top), head line (mid), life line (side)
        finger_base_mid = (idx_mcp + pinky_mcp) / 2.0
        palm_center = wrist * 0.5 + finger_base_mid * 0.5

        # ── Palm width (index → pinky MCP) drives crop size
        palm_width = float(np.linalg.norm(idx_mcp - pinky_mcp))
        crop_size  = int(palm_width * 1.4)   # ~palm width with small margin

        # ── Rotation: align wrist→middle-MCP vector to "straight up" (-Y axis)
        direction   = mid_mcp - wrist
        angle_deg   = float(np.degrees(np.arctan2(direction[0], -direction[1])))

        return self._rotate_and_crop(image, palm_center, angle_deg, crop_size)

    # ───────────────────────── fallback: contour-based (dark-background scans)
    def _extract_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Otsu threshold works well for dark-bg dataset images
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clean noise
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        rect    = cv2.minAreaRect(largest)      # (center, (w, h), angle)
        center, (bw, bh), angle = rect

        # Use the shorter axis as crop size, keep 75% of it (trim edges/wrist)
        short_side = min(bw, bh)
        crop_size  = int(short_side * 0.75)

        return self._rotate_and_crop(
            image, np.array(center, dtype=np.float32), angle, crop_size
        )

    # ─────────────────────────────────────────────── shared geometric helper
    def _rotate_and_crop(
        self,
        image: np.ndarray,
        center: np.ndarray,
        angle_deg: float,
        crop_size: int,
    ) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        cx, cy = float(center[0]), float(center[1])

        # Rotate image around palm center
        M       = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        half = crop_size // 2
        x1, y1 = int(cx) - half, int(cy) - half
        x2, y2 = x1 + crop_size, y1 + crop_size

        # Shift crop region to stay within image bounds (no black padding)
        if x1 < 0:
            x1, x2 = 0, crop_size
        if y1 < 0:
            y1, y2 = 0, crop_size
        if x2 > w:
            x1, x2 = w - crop_size, w
        if y2 > h:
            y1, y2 = h - crop_size, h

        # Hard-clamp as last resort (image smaller than crop_size)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = rotated[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        roi = cv2.resize(roi, (self.output_size, self.output_size),
                         interpolation=cv2.INTER_AREA)

        if self.grayscale:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        return roi


# ────────────────────────────────────────────────────────────────────────────
def visualise(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """
    Return a side-by-side debug visualisation (original resized | ROI).
    Handy for quick inspection during development.
    """
    thumb_h = 300
    scale   = thumb_h / image.shape[0]
    thumb   = cv2.resize(image, (int(image.shape[1] * scale), thumb_h))

    roi_vis = roi.copy()
    if roi_vis.ndim == 2:
        roi_vis = cv2.cvtColor(roi_vis, cv2.COLOR_GRAY2BGR)
    roi_vis = cv2.resize(roi_vis, (thumb_h, thumb_h))

    return np.hstack([thumb, roi_vis])


# ────────────────────────────────────────────────────────────────────────────
def process_dataset(
    dataset_json: str | Path,
    output_dir:   str | Path,
    output_size:  int = 128,
    split: str | None = None,   # "train" | "test" | None (all)
    save_vis:     bool = False,
):
    """
    Batch-process the dataset described in dataset_split.json.

    Output structure
    ----------------
    output_dir/
      train/
        {person_id:04d}/
          session1/00001.png ...
          session2/00001.png ...
      test/
        ...
    """
    base = Path(dataset_json).parent
    with open(dataset_json) as f:
        data = json.load(f)

    extractor = PalmROIExtractor(output_size=output_size, grayscale=True)
    out_root  = Path(output_dir)

    persons = data["persons"]
    if split:
        persons = [p for p in persons if p["split"] == split]

    total = sum(len(p["session1"]) + len(p["session2"]) for p in persons)
    ok = fail = done = 0

    for person in persons:
        pid = person["person_id"]
        sp  = person["split"]

        for session_name in ["session1", "session2"]:
            src_dir = base / "Dataset" / session_name
            out_dir = out_root / sp / f"{pid:04d}" / session_name
            out_dir.mkdir(parents=True, exist_ok=True)

            if save_vis:
                vis_dir = out_root / "vis" / sp / f"{pid:04d}" / session_name
                vis_dir.mkdir(parents=True, exist_ok=True)

            for fname in person[session_name]:
                src_path = src_dir / fname
                img = cv2.imread(str(src_path))
                done += 1

                if img is None:
                    print(f"  [WARN] cannot read {src_path}")
                    fail += 1
                    continue

                roi = extractor.extract(img)
                if roi is None:
                    print(f"  [FAIL] ROI extraction failed: {src_path}")
                    fail += 1
                    continue

                out_name = Path(fname).stem + ".png"
                cv2.imwrite(str(out_dir / out_name), roi)
                ok += 1

                if save_vis:
                    cv2.imwrite(str(vis_dir / out_name), visualise(img, roi))

                if done % 500 == 0 or done == total:
                    print(f"  [{done}/{total}]  ok={ok}  fail={fail}")

    print(f"\nDone — success: {ok} | failed: {fail}")


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # ── quick single-image test ──────────────────────────────────────────────
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        extractor = PalmROIExtractor(output_size=128, grayscale=True)
        img = cv2.imread(path)
        if img is None:
            print(f"Cannot read: {path}")
            sys.exit(1)

        roi = extractor.extract(img)
        if roi is None:
            print("ROI extraction failed.")
            sys.exit(1)

        vis = visualise(img, roi)
        cv2.imshow("Palm ROI  (left: original  |  right: ROI)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ── batch dataset processing ─────────────────────────────────────────────
    else:
        base = Path(__file__).parent
        process_dataset(
            dataset_json = base / "dataset_split.json",
            output_dir   = base / "ROI_output",
            output_size  = 128,
            save_vis     = False,
        )
