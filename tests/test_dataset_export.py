from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from crossmedia_pid.dataset import export_track_dataset


def _write_tiny_video(path: Path, frame_count: int = 6) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5,
        (64, 64),
    )
    assert writer.isOpened()

    for index in range(frame_count):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.rectangle(frame, (8 + index, 10), (32 + index, 54), (40, 180, 240), -1)
        writer.write(frame)

    writer.release()


def _track(track_id: int, frames: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        track_id=track_id,
        frame_history=frames,
        bbox_history=[(8, 10, 34, 54) for _ in frames],
        confidence_history=[0.8 for _ in frames],
        position_history=["left", "center", "right"][: len(frames)],
        best_frame=frames[0],
        best_bbox=(8, 10, 34, 54),
        best_confidence=0.8,
        best_quality=0.75,
        frame_width=64,
        frame_height=64,
    )


def test_export_track_dataset_creates_manifest_pairs_and_zip(tmp_path):
    video_path = tmp_path / "source.avi"
    _write_tiny_video(video_path)

    tracks = {
        0: _track(0, [1, 3, 5]),
        1: _track(1, [2, 4, 6]),
    }

    result = export_track_dataset(
        video_path=video_path,
        tracks=tracks,
        selected_track_ids=[0, 1],
        dataset_name="demo dataset",
        output_root=tmp_path / "datasets",
        samples_per_track=2,
        negative_pairs_per_identity=1,
    )

    assert result.image_count == 4
    assert result.identity_count == 2
    assert result.positive_pair_count == 2
    assert result.negative_pair_count == 1
    assert result.manifest_path.exists()
    assert result.pairs_path.exists()
    assert result.summary_path.exists()
    assert result.zip_path.exists()
    assert len(list((result.dataset_dir / "images").glob("*.jpg"))) == 4
