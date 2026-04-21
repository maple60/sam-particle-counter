import csv
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import glasbey
from napari.layers import Image, Labels


class ExportService:
    def __init__(self):
        self.palette_glasbey = glasbey.create_palette(
            palette_size=12, colorblind_safe=True
        )

    def _hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            raise ValueError(f"Invalid hex color: {hex_color}")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    def _label_colormap(self, label: int) -> tuple[int, int, int]:
        if label <= 0:
            return (0, 0, 0)  # Transparent for background or non-positive labels
        hex_color = self.palette_glasbey[(label - 1) % len(self.palette_glasbey)]
        return self._hex_to_rgb(hex_color)

    """
    def _label_colormap(self, label: int) -> tuple[int, int, int]:
        seed = int(label * 1103515245 + 12345) & 0x7FFFFFFF
        b = 80 + (seed % 156)
        g = 80 + ((seed // 97) % 156)
        r = 80 + ((seed // 197) % 156)
        return int(r), int(g), int(b)
    """

    def _build_overlay_with_ids(
        self, image_rgb: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        overlay = image_rgb.copy()
        blended = image_rgb.copy()
        unique_labels = [int(v) for v in np.unique(labels) if int(v) > 0]

        for label_id in unique_labels:
            mask = labels == label_id
            if not np.any(mask):
                continue
            color = self._label_colormap(label_id)
            blended[mask] = color

        overlay = cv2.addWeighted(blended, 0.45, image_rgb, 0.55, 0.0)

        for blob in self._compute_blob_metrics(labels):
            cx = int(round(blob["centroid_x"]))
            cy = int(round(blob["centroid_y"]))
            cv2.putText(
                overlay,
                str(blob["id"]),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                str(blob["id"]),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return overlay

    def _build_roi_overlay(
        self, source_rgb: np.ndarray, roi_bbox_yx: list[int] | None
    ) -> np.ndarray:
        canvas = source_rgb.copy()
        if roi_bbox_yx is None:
            return canvas
        ymin, ymax, xmin, xmax = roi_bbox_yx
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (255, 140, 0), 3)
        cv2.putText(
            canvas,
            "ROI",
            (xmin, max(ymin - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 140, 0),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _compute_blob_metrics(self, labels: np.ndarray) -> list[dict[str, Any]]:
        blobs: list[dict[str, Any]] = []
        unique_labels = [int(v) for v in np.unique(labels) if int(v) > 0]
        for label_id in unique_labels:
            mask = (labels == label_id).astype(np.uint8)
            area = int(mask.sum())
            if area == 0:
                continue
            ys, xs = np.where(mask > 0)
            y_min = int(ys.min())
            y_max = int(ys.max())
            x_min = int(xs.min())
            x_max = int(xs.max())
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            perimeter = float(sum(cv2.arcLength(cnt, True) for cnt in contours))
            circularity = (
                float((4.0 * np.pi * area) / (perimeter * perimeter))
                if perimeter > 0.0
                else float("nan")
            )
            moments = cv2.moments(mask)
            if moments["m00"] == 0:
                cx = float(xs.mean())
                cy = float(ys.mean())
            else:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])

            blobs.append(
                {
                    "id": label_id,
                    "area_px": area,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "bbox_x_min": x_min,
                    "bbox_y_min": y_min,
                    "bbox_x_max": x_max,
                    "bbox_y_max": y_max,
                    "bbox_width": x_max - x_min + 1,
                    "bbox_height": y_max - y_min + 1,
                    "perimeter_px": perimeter,
                    "circularity": circularity,
                }
            )
        return blobs

    def _write_blob_csv(self, path: Path, labels: np.ndarray) -> None:
        rows = self._compute_blob_metrics(labels)
        fieldnames = [
            "id",
            "area_px",
            "centroid_x",
            "centroid_y",
            "bbox_x_min",
            "bbox_y_min",
            "bbox_x_max",
            "bbox_y_max",
            "bbox_width",
            "bbox_height",
            "perimeter_px",
            "circularity",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _count_positive_labels(self, labels: np.ndarray) -> int:
        unique_labels = np.unique(np.asarray(labels, dtype=np.int32))
        return int(np.count_nonzero(unique_labels > 0))

    def _build_export_metadata(
        self,
        image_name: str,
        source_layer: Image,
        cropped_layer: Image,
        sam2_layer: Labels,
        final_layer: Labels,
        sam2_labels: np.ndarray,
        final_labels: np.ndarray,
        roi_bbox_yx: list[int] | None,
        output_files: list[str],
        sam2_checkpoint_cfg_getter: Callable[[], tuple[str, str]],
    ) -> dict[str, Any]:
        sam2_payload = dict(getattr(sam2_layer, "metadata", {}) or {}).get(
            "male_flower_count", {}
        )
        sam2_params = sam2_payload.get("sam2_params", {})
        dedup_removed = int(sam2_payload.get("dedup_removed_count", 0))
        sam2_after_dedup = self._count_positive_labels(sam2_labels)
        sam2_raw = sam2_after_dedup + dedup_removed
        final_count = self._count_positive_labels(final_labels)
        manual_net = final_count - sam2_after_dedup

        return {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_image_layer": image_name,
            "source_image_shape_hw": list(source_layer.data.shape[:2]),
            "cropped_layer_name": cropped_layer.name,
            "cropped_shape_hw": list(cropped_layer.data.shape[:2]),
            "roi_bbox_yx": roi_bbox_yx,
            "sam2_model": {
                "checkpoint_and_cfg": sam2_checkpoint_cfg_getter(),
                "sam2_auto_params": sam2_params,
            },
            "counts": {
                "sam2_raw_count_before_dedup": sam2_raw,
                "sam2_after_dedup_count": sam2_after_dedup,
                "dedup_removed_count": dedup_removed,
                "final_count": final_count,
                "manual_delta_count_net": manual_net,
                "manual_added_estimated": max(0, manual_net),
                "manual_removed_estimated": max(0, -manual_net),
            },
            "layers_used": {
                "sam2_layer": sam2_layer.name,
                "final_layer": final_layer.name,
            },
            "randomness": {
                "python_random_seed_explicitly_set": False,
                "numpy_random_seed_explicitly_set": False,
                "python_random_state_head": list(random.getstate()[1][:5]),
                "numpy_random_state_head": [
                    int(v) for v in np.random.get_state()[1][:5]
                ],
            },
            "runtime_environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "cwd": str(Path.cwd()),
            },
            "output_files": output_files,
        }

    def ensure_export_dir(self, output_dir: Path, image_name: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_name = "".join(
            ch if ch.isalnum() or ch in "-._" else "_" for ch in image_name
        )
        export_dir = Path(output_dir).expanduser() / f"{safe_name}_{timestamp}"
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir

    def extract_latest_roi_bbox(
        self, roi_layer, image_shape: tuple[int, int]
    ) -> list[int] | None:
        if len(roi_layer.data) == 0:
            return None

        roi = np.asarray(roi_layer.data[-1])
        ymin = int(np.floor(roi[:, 0].min()))
        ymax = int(np.ceil(roi[:, 0].max()))
        xmin = int(np.floor(roi[:, 1].min()))
        xmax = int(np.ceil(roi[:, 1].max()))

        h, w = image_shape
        ymin = max(0, min(ymin, h))
        ymax = max(0, min(ymax, h))
        xmin = max(0, min(xmin, w))
        xmax = max(0, min(xmax, w))
        if ymin >= ymax or xmin >= xmax:
            return None
        return [ymin, ymax, xmin, xmax]
