from pathlib import Path
from typing import Any

import cv2
import numpy as np
from napari.layers import Image, Points
from napari.utils.notifications import show_info


class Sam2Service:
    def __init__(self) -> None:
        self._sam2_mask_generators: dict[tuple[int, float, float, int, int], Any] = {}
        self._sam2_image_predictor: Any = None

    def prepare_rgb_image(self, layer: Image) -> np.ndarray:
        image = np.asarray(layer.data)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("Please provide an RGB(A) image.")

        rgb = np.asarray(image[..., :3])
        if rgb.dtype == np.uint8:
            return rgb

        rgb_float = np.asarray(rgb, dtype=np.float32)
        rgb_float = np.nan_to_num(rgb_float, nan=0.0, posinf=255.0, neginf=0.0)

        data_min = float(np.min(rgb_float))
        data_max = float(np.max(rgb_float))

        if data_max <= 1.0 and data_min >= 0.0:
            scaled = rgb_float * 255.0
        elif data_max == data_min:
            scaled = np.zeros_like(rgb_float, dtype=np.float32)
        else:
            scaled = (rgb_float - data_min) * (255.0 / (data_max - data_min))

        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

    def _get_sam2_checkpoint_and_cfg(self) -> tuple[str, str]:
        project_root = Path(__file__).resolve().parents[2]
        checkpoint = (
            project_root / "sam2_repo" / "checkpoints" / "sam2.1_hiera_large.pt"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                "SAM2 checkpoint not found. Run setup equivalent to scripts/setup_sam2.bat first."
            )
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        return str(checkpoint), model_cfg

    def _get_sam2_mask_generator(
        self,
        points_per_side: int,
        pred_iou_thresh: float,
        stability_score_thresh: float,
        crop_n_layers: int,
        crop_n_points_downscale_factor: int,
    ):
        key = (
            int(points_per_side),
            float(pred_iou_thresh),
            float(stability_score_thresh),
            int(crop_n_layers),
            int(crop_n_points_downscale_factor),
        )
        if key in self._sam2_mask_generators:
            return self._sam2_mask_generators[key]

        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        checkpoint, model_cfg = self._get_sam2_checkpoint_and_cfg()

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        show_info(f"using device for SAM2 auto: {device}")

        sam2_model = build_sam2(
            model_cfg, checkpoint, device=device, apply_postprocessing=False
        )
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=key[0],
            pred_iou_thresh=key[1],
            stability_score_thresh=key[2],
            crop_n_layers=key[3],
            crop_n_points_downscale_factor=key[4],
            box_nms_thresh=0.55,
            crop_nms_thresh=0.55,
            multimask_output=False,
        )
        self._sam2_mask_generators[key] = mask_generator
        return mask_generator

    def _get_sam2_image_predictor(self):
        if self._sam2_image_predictor is not None:
            return self._sam2_image_predictor

        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint, model_cfg = self._get_sam2_checkpoint_and_cfg()

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        show_info(f"using device for SAM2 predictor: {device}")

        sam2_model = build_sam2(
            model_cfg, checkpoint, device=device, apply_postprocessing=False
        )
        self._sam2_image_predictor = SAM2ImagePredictor(sam2_model)
        return self._sam2_image_predictor

    def _masks_to_label_image(
        self, masks: list[dict], image_shape: tuple[int, int]
    ) -> np.ndarray:
        label_image = np.zeros(image_shape, dtype=np.int32)
        for idx, ann in enumerate(masks, start=1):
            mask = ann.get("segmentation")
            if mask is None:
                continue
            label_image[np.asarray(mask, dtype=bool)] = idx
        return label_image

    def _masks_to_polygons(self, masks: list[dict]) -> tuple[list[np.ndarray], list[float]]:
        polygons: list[np.ndarray] = []
        polygon_scores: list[float] = []
        for ann in masks:
            mask = ann.get("segmentation")
            if mask is None:
                continue

            score = float(ann.get("predicted_iou", np.nan))
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if contour.shape[0] < 3:
                    continue
                contour_xy = contour[:, 0, :]
                polygons.append(contour_xy[:, [1, 0]])
                polygon_scores.append(score)
        return polygons, polygon_scores

    def _sort_masks_by_position(self, masks: list[dict], method: str) -> list[dict]:
        def sort_key(ann: dict) -> tuple[float, float]:
            if method == "bbox":
                bbox = ann.get("bbox")
                if bbox is None or len(bbox) < 2:
                    return (np.inf, np.inf)
                x, y = bbox[0], bbox[1]
                return float(y), float(x)

            if method != "centroid":
                raise ValueError("method must be 'centroid' or 'bbox'")

            mask = ann.get("segmentation")
            if mask is None:
                return (np.inf, np.inf)
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                return (np.inf, np.inf)
            return float(np.mean(ys)), float(np.mean(xs))

        return sorted(masks, key=sort_key)

    def _mask_overlap_stats(self, mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, float]:
        inter = float(np.logical_and(mask_a, mask_b).sum())
        area_a = float(mask_a.sum())
        area_b = float(mask_b.sum())
        union = float(np.logical_or(mask_a, mask_b).sum())
        iou = inter / union if union > 0 else 0.0
        contain = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
        return {
            "intersection": inter,
            "area_a": area_a,
            "area_b": area_b,
            "iou": iou,
            "contain": contain,
        }

    def _deduplicate_masks(
        self,
        masks: list[dict],
        contain_thresh: float = 0.9,
        iou_thresh: float = 0.7,
        score_key: str = "predicted_iou",
    ) -> tuple[list[dict], list[tuple[int, int, dict[str, float]]]]:
        def _score_value(mask: dict) -> float:
            value = mask.get(score_key, 0.0)
            if np.isscalar(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0
            return 0.0

        n_masks = len(masks)
        removed: set[int] = set()
        removed_pairs: list[tuple[int, int, dict[str, float]]] = []

        for i in range(n_masks):
            if i in removed:
                continue

            mask_i = masks[i].get("segmentation")
            if mask_i is None:
                continue

            for j in range(i + 1, n_masks):
                if j in removed:
                    continue

                mask_j = masks[j].get("segmentation")
                if mask_j is None:
                    continue

                stats = self._mask_overlap_stats(mask_i, mask_j)
                if stats["contain"] < contain_thresh and stats["iou"] < iou_thresh:
                    continue

                score_i = _score_value(masks[i])
                score_j = _score_value(masks[j])

                if score_i >= score_j:
                    removed.add(j)
                    removed_pairs.append((i, j, stats))
                else:
                    removed.add(i)
                    removed_pairs.append((j, i, stats))
                    break

        kept_masks = [m for idx, m in enumerate(masks) if idx not in removed]
        return kept_masks, removed_pairs

    def _collect_point_prompts(
        self, image_layer: Image, layers
    ) -> tuple[np.ndarray, np.ndarray]:
        fg_points_name = f"{image_layer.name}_prompts_fg"
        bg_points_name = f"{image_layer.name}_prompts_bg"
        legacy_points_name = f"{image_layer.name}_prompts"

        layers_coords_xy: list[np.ndarray] = []
        layers_labels: list[np.ndarray] = []

        def _append_points(points_layer: Points, fixed_label: int | None = None) -> None:
            if len(points_layer.data) == 0:
                return

            labels = np.ones((len(points_layer.data),), dtype=np.int32)
            if fixed_label is not None:
                labels.fill(fixed_label)
            elif "label" in points_layer.features:
                labels = np.asarray(points_layer.features["label"], dtype=np.int32)

            valid = np.isin(labels, [0, 1])
            if not np.all(valid):
                invalid_values = np.unique(labels[~valid])
                raise ValueError(
                    f"Labels can only be 0 or 1. Invalid values: {invalid_values.tolist()}"
                )

            coords_yx = np.asarray(points_layer.data)[:, :2]
            coords_xy = coords_yx[:, [1, 0]].astype(np.float32)
            layers_coords_xy.append(coords_xy)
            layers_labels.append(labels)

        has_fg = fg_points_name in layers
        has_bg = bg_points_name in layers
        has_legacy = legacy_points_name in layers

        if has_fg:
            fg_layer = layers[fg_points_name]
            if not isinstance(fg_layer, Points):
                raise ValueError(f"{fg_points_name} is not a Points layer.")
            _append_points(fg_layer, fixed_label=1)

        if has_bg:
            bg_layer = layers[bg_points_name]
            if not isinstance(bg_layer, Points):
                raise ValueError(f"{bg_points_name} is not a Points layer.")
            _append_points(bg_layer, fixed_label=0)

        if has_legacy:
            points_layer = layers[legacy_points_name]
            if not isinstance(points_layer, Points):
                raise ValueError(
                    f"{legacy_points_name} is not a Points layer."
                )
            _append_points(points_layer)

        if not (has_fg or has_bg or has_legacy):
            raise ValueError(
                f"{fg_points_name} / {bg_points_name} / {legacy_points_name} not found. Create them with Create prompt points layer."
            )

        if not layers_coords_xy:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)

        return (
            np.concatenate(layers_coords_xy, axis=0).astype(np.float32),
            np.concatenate(layers_labels, axis=0).astype(np.int32),
        )
