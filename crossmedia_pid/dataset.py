"""Dataset export helpers for tracked person crops."""

from __future__ import annotations

import csv
import itertools
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2

from crossmedia_pid.config import project_path


@dataclass
class DatasetExportResult:
    """Result metadata for a generated test dataset."""

    dataset_dir: Path
    zip_path: Path
    manifest_path: Path
    pairs_path: Path
    summary_path: Path
    image_count: int
    identity_count: int
    positive_pair_count: int
    negative_pair_count: int


def slugify(value: str, fallback: str = "dataset") -> str:
    """Return a filesystem-safe ASCII slug."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = value.strip("._-")
    return value or fallback


def default_dataset_name(prefix: str = "video_testset") -> str:
    """Create a timestamped dataset name."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def default_dataset_root() -> Path:
    """Default location for generated experiment datasets."""
    return project_path("experiments", "generated_datasets")


def _sample_indices(total: int, samples_per_track: int) -> list[int]:
    if total <= 0 or samples_per_track <= 0:
        return []
    if total <= samples_per_track:
        return list(range(total))
    if samples_per_track == 1:
        return [total // 2]

    step = (total - 1) / (samples_per_track - 1)
    indices = [round(i * step) for i in range(samples_per_track)]
    return sorted(set(min(total - 1, max(0, index)) for index in indices))


def _read_frame(cap: cv2.VideoCapture, frame_number: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 1))
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Unable to read frame {frame_number}")
    return frame


def _crop_frame(frame, bbox: tuple[int, int, int, int], padding: float):
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_pairs(records: list[dict[str, Any]], negative_pairs_per_identity: int) -> list[dict[str, Any]]:
    by_person: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_person.setdefault(record["person_id"], []).append(record)

    pairs: list[dict[str, Any]] = []
    pair_index = 0
    for person_records in by_person.values():
        for left, right in itertools.combinations(person_records, 2):
            pairs.append(
                {
                    "pair_id": f"pair_{pair_index:06d}",
                    "image_a": left["image_path"],
                    "image_b": right["image_path"],
                    "label": 1,
                    "person_a": left["person_id"],
                    "person_b": right["person_id"],
                }
            )
            pair_index += 1

    identities = sorted(by_person)
    for left_person, right_person in itertools.combinations(identities, 2):
        left_records = by_person[left_person]
        right_records = by_person[right_person]
        limit = min(negative_pairs_per_identity, len(left_records), len(right_records))
        for index in range(limit):
            left = left_records[index % len(left_records)]
            right = right_records[-(index % len(right_records)) - 1]
            pairs.append(
                {
                    "pair_id": f"pair_{pair_index:06d}",
                    "image_a": left["image_path"],
                    "image_b": right["image_path"],
                    "label": 0,
                    "person_a": left["person_id"],
                    "person_b": right["person_id"],
                }
            )
            pair_index += 1

    return pairs


def _zip_dataset(dataset_dir: Path) -> Path:
    zip_path = dataset_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dataset_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(dataset_dir.parent))

    return zip_path


def export_track_dataset(
    *,
    video_path: Path,
    tracks: dict[int, Any],
    selected_track_ids: list[int],
    dataset_name: str,
    output_root: Path | None = None,
    samples_per_track: int = 4,
    padding: float = 0.1,
    min_track_frames: int = 2,
    negative_pairs_per_identity: int = 2,
) -> DatasetExportResult:
    """Export selected tracked identities as a crop-based test dataset."""
    if output_root is None:
        output_root = default_dataset_root()

    dataset_name = slugify(dataset_name, fallback=default_dataset_name())
    dataset_dir = output_root / dataset_name
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    selected = [track_id for track_id in selected_track_ids if track_id in tracks]
    if not selected:
        raise ValueError("No valid tracks selected")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    try:
        person_index = 0
        for track_id in selected:
            track = tracks[track_id]
            if len(track.frame_history) < min_track_frames:
                continue

            person_id = f"p{person_index:04d}"
            person_index += 1
            indices = _sample_indices(len(track.frame_history), samples_per_track)

            for sample_index, history_index in enumerate(indices):
                frame_number = track.frame_history[history_index]
                bbox = tuple(int(value) for value in track.bbox_history[history_index])
                confidence = float(track.confidence_history[history_index])
                position = (
                    track.position_history[history_index]
                    if hasattr(track, "position_history")
                    else "unknown"
                )

                frame = _read_frame(cap, frame_number)
                crop, padded_bbox = _crop_frame(frame, bbox, padding)
                filename = (
                    f"{person_id}_track{track_id:03d}_"
                    f"f{frame_number:06d}_{position}_{sample_index:02d}.jpg"
                )
                image_path = images_dir / filename
                cv2.imwrite(str(image_path), crop)

                rel_path = image_path.relative_to(dataset_dir).as_posix()
                records.append(
                    {
                        "image_path": rel_path,
                        "person_id": person_id,
                        "track_id": track_id,
                        "sample_index": sample_index,
                        "frame_number": frame_number,
                        "position": position,
                        "confidence": round(confidence, 6),
                        "track_frame_count": len(track.frame_history),
                        "track_best_frame": track.best_frame,
                        "track_best_quality": round(float(track.best_quality), 6),
                        "bbox_x1": padded_bbox[0],
                        "bbox_y1": padded_bbox[1],
                        "bbox_x2": padded_bbox[2],
                        "bbox_y2": padded_bbox[3],
                        "source_video": str(video_path),
                    }
                )
    finally:
        cap.release()

    if not records:
        raise ValueError("Selected tracks did not produce any samples")

    pairs = _build_pairs(records, negative_pairs_per_identity)
    positive_pair_count = sum(1 for pair in pairs if pair["label"] == 1)
    negative_pair_count = sum(1 for pair in pairs if pair["label"] == 0)

    manifest_path = dataset_dir / "manifest.csv"
    manifest_jsonl_path = dataset_dir / "manifest.jsonl"
    pairs_path = dataset_dir / "pairs.csv"
    summary_path = dataset_dir / "summary.json"

    _write_csv(manifest_path, records)
    _write_jsonl(manifest_jsonl_path, records)
    _write_csv(pairs_path, pairs)

    summary = {
        "dataset_name": dataset_name,
        "source_video": str(video_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "identity_count": len({record["person_id"] for record in records}),
        "track_count": len({record["track_id"] for record in records}),
        "image_count": len(records),
        "positive_pair_count": positive_pair_count,
        "negative_pair_count": negative_pair_count,
        "samples_per_track": samples_per_track,
        "padding": padding,
        "min_track_frames": min_track_frames,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = _zip_dataset(dataset_dir)
    return DatasetExportResult(
        dataset_dir=dataset_dir,
        zip_path=zip_path,
        manifest_path=manifest_path,
        pairs_path=pairs_path,
        summary_path=summary_path,
        image_count=len(records),
        identity_count=summary["identity_count"],
        positive_pair_count=positive_pair_count,
        negative_pair_count=negative_pair_count,
    )
